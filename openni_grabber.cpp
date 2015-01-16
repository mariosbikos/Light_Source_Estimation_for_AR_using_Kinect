
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

#include <GL/glut.h>

#include <Eigen/SVD>
#include <Eigen/Core>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/visualization/keyboard_event.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/time.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/filesystem.hpp>

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <math.h>

#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */ 
using namespace std;

#define PI 3.14
//Speed of interaction(user-object)
#define MANIPULATION_SPEED 2.0
#define NORMALS 1
#define SEGMENTATION 2

//Pointer of Cloud of xyzPoints
typedef pcl::PointXYZRGBA								Point;
//Constant Pointer of Cloud of xyzrgbaPoints
typedef pcl::PointCloud<Point>::ConstPtr				PointCloudCPtr;
//Pointer of Cloud of xyzrgbaPoints
typedef pcl::PointCloud<Point>::Ptr						PointCloudPtr;
//Pointer of Cloud of Normals
typedef pcl::PointCloud<pcl::Normal>::Ptr				PointCloudNormalPtr;
//Constant Pointer of Point Cloud of Normals
typedef pcl::PointCloud<pcl::Normal>::ConstPtr			PointCloudNormalCPtr;


//---Variables for OpenGL object Manipulation---//
static float tx = 0.0,ty=0.0,tz=0.0;
static float rotx = 0.0,roty=0.0,rotz=0.0;
int xflag=1,zflag=1,yflag=1;
int old_x=0;
int old_y=0;
int valid=0;
int dx=1.0,dy=1.0;
double scale1x=1.0,scale1y=1.0,scale1z=1.0;
double total=1.0;

//----------------------------------------------//

//--Flags for OpenGL different renderings(Normals,Segmented Colored Cloud,etc)
int enable_normals_draw=0;
int enable_colored_cloud_draw=0;
int enable_cloud_draw=1;
int enable_draw_subclouds=0;
int enable_light_draw=0;
int enable_lightSource_draw=0;

//--Variables for SubCloud Selection------------//
int totalClusters=0;
int currentCluster=0;
vector< vector<int> > indiciesAll;
vector< vector<int> > indicies_used;

//--Public PointClouds used
PointCloudPtr points;
PointCloudNormalPtr normals_estimated;
PointCloudPtr colored_cloud;

//--Eigen's Vectors for light source position-------------//
Eigen::Vector3f ls;
Eigen::Vector3f totalPosition;

double realx=0,realy=0,realz=0;
double length=0;
double angle=0,anglexy=0;
//______________________________________________________________________________//


//Class Recorder
//Scannarei ton xoro kai dimiourgei ena pcd arxeio me ola ta simeia pou anagnoristikan apo to kinect
//To arxeio exei mesa x,y,z kai rgba(kentro ton aksonon to kentro tis othonis (0,0,z)
class Recorder
{
public:
	Recorder(void):_counts(0){}
	~Recorder(void){}

	void run()
	{
	   pcl::Grabber* interface = new pcl::OpenNIGrabber();
	   boost::function<void (const PointCloudCPtr&)> f = boost::bind (&Recorder::_callback, this, _1);
       interface->registerCallback (f);
       interface->start ();
	   if(_counts<1)
	   {
		   boost::this_thread::sleep (boost::posix_time::seconds (1));
	   }
	   interface->stop();
	}
private:
	int _counts;

	void _callback(const PointCloudCPtr &cloud)
	{
		cout<<"saving... ";
		cout<<"checking the cloud"<<endl;
		//Orizo ena neo cloud
		PointCloudPtr validcloud( new pcl::PointCloud<Point>() );
		//vector me integers
		vector<int> changes;
		//Removes points with x,y,z equal to NaN
		//To validcloud einai opos to cloud alla afairountai osa simeia eixan metrisi NaN
		//O arithmos ton allagon einai to mikos tou vector changes
		pcl::removeNaNFromPointCloud<Point>(*cloud,*validcloud,changes);
		cout << "NaN points removed: " << (cloud->size())-(validcloud->size()) << endl;

		//Change upside down cloud to correct position and then save to .pcd file
		PointCloudPtr deep_cloud (new pcl::PointCloud<Point>( *validcloud ) );
		/*for (size_t i = 0; i < deep_cloud->points.size (); ++i)
		{   
		  deep_cloud->points[i].y = -deep_cloud->points[i].y;
		  deep_cloud->points[i].z = -deep_cloud->points[i].z;
		}*/
		//PointCloudCPtr correct_cloud = deep_cloud;
		PointCloudCPtr correct_cloud = validcloud;
		pcl::io::savePCDFile<Point>(boost::lexical_cast<string>(_counts)+ ".pcd" , *correct_cloud);
		cout<<"File 0.pcd saved"<<endl;
		_counts++;
	}

};

//______________________________________________________________________________//

//Class Loader
//Metatrepei to 0.pcd se binary 0.b.pcd kai periexei ti methodo Loader().load() pou epistrefei to cloud
//eite se binary eite se ascii morfi
class Loader
{
public:
	Loader(void){}
	~Loader(void){}

	PointCloudPtr load()
	{
		cout<<"Loading PCD file "<<endl;
		PointCloudPtr cloud(new pcl::PointCloud<Point>());
		cloud->resize(640*480);

		int success;

		if(boost::filesystem::exists("0.b.pcd"))
		{
			cout<<"loading Binary Version"<<endl;
			success = pcl::io::loadPCDFile<Point>("0.b.pcd",*cloud);
		}
		else
		{
			cout<<"Loading ascii version"<<endl;
			success = pcl::io::loadPCDFile<Point>("0.pcd",*cloud);
		}
		if(success<0)
			throw runtime_error("failed to load pcd");

		return cloud;
	}


	void binarize()
	{

		if(boost::filesystem::exists("0.b.pcd"))
		{
			cout<<"Already Binarized"<<endl;
			return;
		}

		points=load();

		cout<<"Converting to Binary "<<endl;
		int success=pcl::io::savePCDFileBinary<Point>("0.b.pcd",*points);
		if(success<0)
			throw runtime_error("failed to load pcd ");
	}
};

//_________________________________________________________________________________________________________________//

//Class NormalsEstimation
//Checks if there is any available 0.n.b.pcd file,otherwise estimates new normals
class NormalsEstimation
{
public:
	NormalsEstimation(void){}
	~NormalsEstimation(void){}

	PointCloudNormalPtr run(PointCloudCPtr cloud)
	{
		cout<<"surface normal estimation of " << cloud->size() << " points" << endl;
		//normals = cloud of normals pointer
		PointCloudNormalPtr normals(new pcl::PointCloud<pcl::Normal>);

		//Check if there is a file with normals already calculated
		if (boost::filesystem::exists("0.n.b.pcb"))
		{
			cout<<"trying to load from cache" << endl;
			int success = pcl::io::loadPCDFile<pcl::Normal>("0.n.b.pcb", *normals);
			if (success >= 0)
			{
				cout<<"successfully loaded from cache" << endl;
				return normals;
			}
		}
		

		//Create normal estimation object(pcl)
		pcl::NormalEstimation<Point, pcl::Normal> normalEstimationObject;
		normalEstimationObject.setInputCloud (cloud);
		//set search method properties
		pcl::search::KdTree<Point>::Ptr tree(new pcl::search::KdTree<Point>);
		normalEstimationObject.setSearchMethod (tree);
		//Use all neighbours in a 3cm radius sphere
		normalEstimationObject.setRadiusSearch (0.03);
		clock_t startClock2 = clock();
		normalEstimationObject.compute(*normals);
		clock_t endClock2 = clock();
		cout<<"Normal Estimation Execution Time: "<< ((double)(endClock2 - startClock2)/CLOCKS_PER_SEC)/60<<" minutes"<<endl ;
		cout<<"Normals calculated: "<<normals->size()<<endl;
		cout<<"save results"<<endl;

		//save file as 0.n.b.pcd
		pcl::io::savePCDFileBinary<pcl::Normal>("0.n.b.pcb", *normals);

		return normals;
	}

};


//______________________________________________________________________________
 

//Color Region Growing Segmentation of a PointCloud, using indicies as a reference
class CloudSegmentation
{
public:
	CloudSegmentation(void){}
	~CloudSegmentation(void){}

	void segment_Cloud(PointCloudCPtr cloud, PointCloudNormalPtr normals, vector< vector<int> > &indicies)
	{
		cout<<"Check for Color_Cloud PCD file "<<endl;
		PointCloudPtr loaded_cloud(new pcl::PointCloud<Point>());
		//resize to maximum possible capacity of kinect
		loaded_cloud->resize(640*480);

		int success=-1;

		//Check if a segmentation File with Segmentation Results already exists
		if(boost::filesystem::exists("0.seg.txt"))
		{
			cout<<"Loading from cached segmentation..."<<endl;
			ifstream file("0.seg.txt");
			boost::archive::xml_iarchive ia(file);
			ia>>BOOST_SERIALIZATION_NVP(indicies);
			cout<<"Loaded "<<indicies.size()<<" clusters"<<endl;
			totalClusters=indicies.size();
		}
		else
		{
			cout<<"Failed to load Segmentation results file 0.seg.txt"<<endl;
		}

		//Check if a segmentation File with Colored PointCloud already exists(binary or ascii) and load it
		if(boost::filesystem::exists("color_cloud.b.pcd"))
		{
			cout<<"loading Binary Version of colored cloud"<<endl;
			success = pcl::io::loadPCDFile<Point>("color_cloud.b.pcd",*loaded_cloud);
			colored_cloud=loaded_cloud;
		}
		else
		{
			if(boost::filesystem::exists("color_cloud.pcd"))
			{
				cout<<"Loading ascii version"<<endl;
				success = pcl::io::loadPCDFile<Point>("color_cloud.pcd",*loaded_cloud);
				colored_cloud=loaded_cloud;
			}
			
		}

		
		if(success<0)
		{
			cout<<"failed to load colored_cloud pcd"<<endl;

			//Start Segmentation
			cout<<"Computating segmentation..."<<endl;
			pcl::IndicesPtr indices (new vector <int>);
			pcl::PassThrough<Point> pass;
			pass.setInputCloud(cloud);
			//--------Best Kinect Range!-------//
			//pass.setFilterFieldName ("z");
			//pass.setFilterLimits (0.8, 4.0);
			pass.filter(*indices);

			pcl::search::Search <pcl::PointXYZRGBA>::Ptr tree = 
				boost::shared_ptr<pcl::search::Search<pcl::PointXYZRGBA> > (new pcl::search::KdTree<pcl::PointXYZRGBA>);
			
			pcl::RegionGrowingRGB<Point,pcl::Normal> reg;
			reg.setInputCloud(cloud);
			reg.setInputNormals(normals);
			reg.setIndices(indices);
			reg.setSearchMethod(tree);
			reg.setDistanceThreshold(100);
			reg.setPointColorThreshold(4);
			reg.setRegionColorThreshold(8);
			reg.setMinClusterSize(600);
			
			std::vector <pcl::PointIndices> *clusters=new vector<pcl::PointIndices>;
			clock_t startClock = clock();
			reg.extract(*clusters);
			clock_t endClock = clock();
			cout<<"Segmentation Execution Time: "<<( (double)(endClock - startClock)/CLOCKS_PER_SEC )/60<<" minutes"<<endl ;
			cout<<"Found: "<<clusters->size()<<" clusters"<<endl;
			totalClusters=clusters->size();

			colored_cloud = reg.getColoredCloudRGBA();

			//Save to txt the indicies of subclouds
			for(size_t i=0; i<clusters->size(); i++)
			{
				indicies.push_back(clusters->at(i).indices);
			}
			cout<<"Segmentation algorithm output consists of "<<indicies.size()<<" segments(sub_clouds)"<<endl;
			cout<<"Saving segmentation results to file 0.seg.txt"<<endl;
			ofstream file("0.seg.txt");
			boost::archive::xml_oarchive oa(file);
			oa & BOOST_SERIALIZATION_NVP(indicies);
			file.close();
			cout<<"Saved!"<<endl;

			//Save to file the color_cloud
			PointCloudCPtr color_cloud = colored_cloud;
			pcl::io::savePCDFile<Point>("color_cloud.pcd" , *color_cloud);
			pcl::io::savePCDFileBinary<Point>("color_cloud.b.pcd",*color_cloud);
			cout<<"File color_cloud.pcd saved"<<endl;
		}
		
	}
	/*
	double estimateCurvatures(PointCloudCPtr cloud, PointCloudNormalCPtr normals, vector<int> &indicies)
	{
		size_t n=indicies.size();
		double average_curvature=0;
		for(size_t i=0;i<n;i++)
		{
			pcl::Normal nor=normals->at(indicies[i]);
			
			if(nor.curvature == nor.curvature)
				average_curvature+=nor.curvature;

		}
		//cout<<"Average: "<<average_curvature<<"n: "<<n<<endl;
		return average_curvature/n;
	}*/
};


//______________________________________________________________________________


//-Class LSEstimation 
//-Methods for the estimation of Point Light Source Position coordinates 
class LightSourceEstimation
{
	public:
	
	LightSourceEstimation(void){}
	~LightSourceEstimation(void){}

	Eigen::Vector3f runEstimation(PointCloudCPtr cloud,PointCloudNormalPtr normals,vector< vector<int> > &indicies)
	{
		ofstream ofile("0.results.txt");
		cout<<"Starting estimations for the Position of the Light Source..."<<endl;

		totalPosition= Eigen::Vector3f::Zero();
		for(size_t i=0; i<indicies.size(); i++)
		{
			Eigen::Vector3f ls= estimateLightSource(cloud,normals,indicies[i]);
			ofile<<"Segment: "<<i<<endl;
			ofile<<ls<<endl<<endl;

			totalPosition=totalPosition+ls/indicies.size();
		}

		cout<<"Combined: "<<endl<<totalPosition<<endl;
		totalPosition.normalize();
		cout<<"Final: "<<endl<<totalPosition<<endl;
		enable_light_draw=1;
		ofile<<"final: "<<endl;
		ofile<<totalPosition<<endl;
		ofile.close();
		return totalPosition;
	}

	Eigen::Vector3f estimateLightSource(PointCloudCPtr cloud, PointCloudNormalCPtr normals, vector<int> &indicies)
	{
		//n=number of indices in parameter's cluster indicies
		size_t n=indicies.size(); 
		cout<<"Estimating LS location from "<<n<<" points"<<endl;
		const static int limit=7000;
		int times=indicies.size()/limit;

		Eigen::Vector3f ret=Eigen::Vector3f::Zero();

		for(int i=0;i<=times;i++)
		{
			n=indicies.size(); 
			cout<<"subsample iteration: "<<i<<"/"<<times<<endl;
			//too big sub sample
			vector<int> current=indicies;
			random_shuffle(current.begin(),current.end());
			if(n<limit)
			{
				cout<<"resizing down to: "<<n<<endl;
				current.resize(n);
			}
			else
			{
				cout<<"resizing down to: "<<limit<<endl;
				current.resize(limit);
			}
			sort(current.begin(),current.end());
			
			if(n<limit)
				;
			else
				n=limit;

			Eigen::MatrixX3f A(n,3);
			Eigen::VectorXf b(n);

			float totalR=0;

			for(size_t i=0;i<n;i++)
			{
				pcl::Normal n=normals->at(indicies[i]);
				
				A(i,0)=n.normal[0];
				A(i,1)=n.normal[1];
				A(i,2)=n.normal[2];

				Point p=cloud->at(indicies[i]);
				//RGB to Grayscale conversion
				float r=0.2126*p.r + 0.7152*p.g+0.0722*p.b;
				totalR+=r;
				b(i)=r+n.normal[0]*p.x+n.normal[1]*p.y+n.normal[2]*p.z;

			}


			cout<<"SVD..."<<endl;
			Eigen::JacobiSVD<Eigen::MatrixX3f> svd(A,Eigen::ComputeFullU | Eigen::ComputeFullV);
			Eigen::VectorXf L=svd.solve(b);
			L.normalize();

			ret+=L;
		}

		ret.normalize();
		cout<<"Estimated ls: "<<endl<<ret<<endl<<endl;
		return ret;
	}
};

//______________________________________________________________________________



void Keyboard(unsigned char key,int x,int y)
{
	
	switch(key)
	{
	case 32:
			tx = 0.0;
			ty=0.0;
			tz=0.0;
			rotx = 0.0;
			roty=0.0;
			rotz=0.0;
			scale1x=1.0;
			scale1y=1.0;
			scale1z=1.0;
		break;
	case 'c' :
		//User enters the coordinates of real light source's position
		//Afterwards program normalizes the value and draws the light source as a sphere in openGL
			cout<<"Enter the coordinates x y z of the real light source(separated by space): ";
			cin>>realx>>realy>>realz;
			cout<<endl;
			cout<<"You entered: x= "<<realx<<" ,y= "<<realy<<" ,z= "<<realz<<endl;

			length=sqrt(realx * realx + realy * realy + realz * realz);
			realx = realx/length;
			realy=realy/length;
			realz = realz/length;
			cout<<"Normalized vector: "<<realx<<" "<<realy<<" "<<realz<<endl;
			cout<<"Estimating angle and xy angle between 2 vectors..."<<endl;
			angle=acos(realx*totalPosition[0] + realy*totalPosition[1] + realz*totalPosition[2]);
			cout<<"Angle= "<<(angle*180.0)/PI<<endl;
			
			enable_lightSource_draw=1;
		break;

	case 'q' : 
			tz-=MANIPULATION_SPEED; 
		break;
	case 'w' : 
			tz+=MANIPULATION_SPEED; 
		break;
	case 'a' : 
			tx-=MANIPULATION_SPEED; 
		break;
	case 's' : 
			tx+=MANIPULATION_SPEED; 
		break;
	case 'z' : 
			ty-=MANIPULATION_SPEED; 
		break;
	case 'x' : 
			ty+=MANIPULATION_SPEED; 
		break;
	case 'o' : 
			rotx-=MANIPULATION_SPEED; 
		break;
	case 'p' : 
			rotx+=MANIPULATION_SPEED; 
		break;
	case 'k' : 
			roty-=MANIPULATION_SPEED; 
		break;
	case 'l' : 
			roty+=MANIPULATION_SPEED; 
		break;
	case 'n' : 
			rotz-=MANIPULATION_SPEED; 
		break;
	case 'm' : 
			rotz+=MANIPULATION_SPEED; 
		break;
	case 'h' :
		{
			//start Segment Selection Process
			cout<<"Starting Segment Selection Process..."<<endl;

			//shut down other flags
			enable_cloud_draw=0;
			enable_normals_draw=0;
			enable_colored_cloud_draw=0;

			//enable flag to draw subclouds
			enable_draw_subclouds=1;

			break;
		}
	case 't' :

		if(currentCluster+1==totalClusters && enable_cloud_draw==0)
		{
			cout<<"You decided to keep "<<indicies_used.size()<<" segments(sub_clouds)"<<endl;
			cout<<"Saving Selection results"<<endl;
			ofstream file("0.sel.txt");
			boost::archive::xml_oarchive oa(file);
			oa & BOOST_SERIALIZATION_NVP(indicies_used);
			file.close();
			cout<<"Saved!"<<endl;

			enable_cloud_draw=1;
			enable_draw_subclouds=0;
		}
		else
		{
			indicies_used.push_back(indiciesAll.at(currentCluster));
			currentCluster++;
		}
			
		break;
	case 'f' :
		if(currentCluster+1==totalClusters && enable_cloud_draw==0)
		{
			cout<<"You decided to keep "<<indicies_used.size()<<" segments(sub_clouds)"<<endl;
			cout<<"Saving Selection results"<<endl;
			ofstream file("0.sel.txt");
			boost::archive::xml_oarchive oa(file);
			oa & BOOST_SERIALIZATION_NVP(indicies_used);
			file.close();
			cout<<"Saved!"<<endl;
			enable_cloud_draw=1;
			enable_draw_subclouds=0;
		}
		else
			currentCluster++;
		break;
	case 'e' :
		{
		   clock_t startClock4 = clock();
		   ls = LightSourceEstimation().runEstimation(points,normals_estimated,indicies_used);
		   clock_t endClock4 = clock(); 
		   cout<<"Light Estimation Time: "<< ((double)(endClock4 - startClock4)/CLOCKS_PER_SEC)<<"seconds"<<endl ;
		 break;
		}
	case 27: // Escape key
      
      exit (0);
      break;
	default : break;
	}

	glutPostRedisplay();
	//
}


void Mouse(int button,int state,int x,int y)
{
	
	 old_x=x; 
	 old_y=y; 
	 if(state == GLUT_DOWN)
	 {
		 valid=1;
	 }
}

void MouseMotion(int x,int y)
{
	if (valid) 
	{
		dx = old_x - x;
		dy = old_y - y;
		total=sqrt( pow(static_cast<double> (dx),static_cast<double> (2.0)) + pow(static_cast<double> (dy),static_cast<double> (2.0))); 

			if(dx<0)
			{
				scale1x+=total/100.0;
				scale1y+=total/100.0;
				scale1z+=total/100.0;
			}
			else if(dx>0)
			{
				scale1x-=total/100.0;
				scale1y-=total/100.0;
				scale1z-=total/100.0;
			}
	}
	
	glutPostRedisplay();
  
}

void DrawAxes()
{
	 glLineWidth(2);
	  glBegin(GL_LINES);
			glColor3f(1.0,0,0);
			glVertex3f(0.0,0.0,0.0);
			glVertex3f(40.0,0.0,0.0);
			glColor3f(0,1.0,0);
			glVertex3f(0.0,0.0,0.0);
			glVertex3f(0.0,40.0,0.0);
			glColor3f(0,0,1.0);
			glVertex3f(0.0,0.0,0.0);
			glVertex3f(0.0,0.0,100.0);
	  glEnd();
}


 //-------------------------------OPENGL Stuff-----------------------//
 void Render()
{    
   
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  // Clean up the colour of the window
                                                       // and the depth buffer
  glMatrixMode(GL_MODELVIEW); 
  glLoadIdentity();
 
  glColor3f(0.6, 0.6, 0.6);

  glPushMatrix();
	  
      glTranslatef(0, 0, -40.0);
	  
	  //--Manipulation commands--//
	  glTranslatef(tx,ty,tz);
	  glRotatef(rotx,1,0,0);
	  glRotatef(roty,0,1,0);
	  glRotatef(rotz,0,0,1);
	  glScalef(scale1x,scale1y,scale1z);
	  DrawAxes();
	  //-----//
	  //Draw Kinect As a Sphere
	  glPushMatrix(); 
			glColor3f(0.2,0.2,0.8);
			glutSolidSphere(0.05,100,100);
	  glPopMatrix();
	  //An ksekinisei i diadikasia gia drawing of subclouds to choose
		  if(enable_lightSource_draw!=0 )
		  {
			   glPushMatrix(); 
			 // glLoadIdentity();
			  glLineWidth(1.0);
			  glColor3f(0.8,0.2,0.2);
			  glBegin(GL_LINES);					
					glVertex3f(0.0,0.0,0.0);
					glVertex3f(realx,-realy,-realz);
			  glEnd();
				  glTranslatef(realx,-realy,-realz);
				  glColor3f(0.8,0.2,0.2);
				  glutSolidSphere(0.05,100,100);
			  glPopMatrix();

		  }
		  if(enable_cloud_draw!=0 )
		  {
			  //Draw Cloud of Points
			  glBegin(GL_POINTS);
			  for (int i=0; i<points->size(); i++)
				{
					//glColor3f(0.5,normals_estimated->at(i).curvature/(double)max_value,0.0);
					glColor3ub(points->at(i).r,points->at(i).g,points->at(i).b);
					glNormal3f(normals_estimated->at(i).normal_x,-normals_estimated->at(i).normal_y,-normals_estimated->at(i).normal_z);
					glVertex3f(points->at(i).x,-points->at(i).y,-points->at(i).z);
				}
			  glEnd();
		 }
		  if(enable_normals_draw!=0 )
		  {
			  //Draw Normals(for every 10 points with length of 0.01
			  glLineWidth(1.0);
			  glColor3f(1.0,0.0,0.0);
			  glBegin(GL_LINES);
			  for (int i=0; i<normals_estimated->size(); i++)
				{
					if(i%10==0)
					{
						glVertex3f(points->at(i).x,-points->at(i).y,-points->at(i).z);
						glVertex3f(points->at(i).x+0.01*normals_estimated->at(i).normal_x,-(points->at(i).y+0.01*normals_estimated->at(i).normal_y),-(points->at(i).z+0.01*normals_estimated->at(i).normal_z));
					}
			  }
			  glEnd();
		  }

		  if(enable_colored_cloud_draw!=0)
		  {
			  //Draw Cloud of Points
			  glBegin(GL_POINTS);
			  for (int i=0; i<colored_cloud->size(); i++)
				{
					glColor3ub(colored_cloud->at(i).r,colored_cloud->at(i).g,colored_cloud->at(i).b);
					glVertex3f(colored_cloud->at(i).x,-colored_cloud->at(i).y,-colored_cloud->at(i).z);
				}
			  glEnd();
		  }

		  if(enable_draw_subclouds!=0)
		  {

			  //Draw subcloud of parameter
			  glBegin(GL_POINTS);
			  //Draw all the points of subcloud(subcloud changes gradually)
			  for(int i=0;i<indiciesAll[currentCluster].size();i++)
				{
					glColor3ub(points->at(indiciesAll[currentCluster].at(i)).r,points->at(indiciesAll[currentCluster].at(i)).g,points->at(indiciesAll[currentCluster].at(i)).b);
					glVertex3f(points->at(indiciesAll[currentCluster].at(i)).x,-points->at(indiciesAll[currentCluster].at(i)).y,-points->at(indiciesAll[currentCluster].at(i)).z);
				}
			  glEnd();
		  }

		  if(enable_light_draw!=0)
		  {

			  glPushMatrix(); 
			 // glLoadIdentity();
			  glLineWidth(1.0);
			  glColor3f(0.8,0.8,0.2);
			  glBegin(GL_LINES);
			  
					
					glVertex3f(0.0,0.0,0.0);
					glVertex3f(totalPosition[0],-totalPosition[1],-totalPosition[2]);
					
			  
			  glEnd();
				  glTranslatef(totalPosition[0],-totalPosition[1],-totalPosition[2]);
				  glColor3f(0.8,0.8,0.2);
				  glutSolidSphere(0.05,100,100);
			  glPopMatrix();
		  }

  glPopMatrix();

  glutSwapBuffers();             // All drawing commands applied to the 
                                 // hidden buffer, so now, bring forward
                                 // the hidden buffer and hide the visible one
}

 void process_F_Keys(int key, int x, int y) 
{
     switch (key) 
    {    
	   //Enable/Disable normals draw
       case GLUT_KEY_F1 :  
		   if(enable_normals_draw!=0)
				enable_normals_draw=0;
		   else if(enable_normals_draw==0)
			    enable_normals_draw=1;
		   break;
		//Enable/Disable Segmentation results
	   case GLUT_KEY_F2 :
		   if(enable_colored_cloud_draw!=0)
				enable_colored_cloud_draw=0;
		   else if(enable_colored_cloud_draw==0)
			    enable_colored_cloud_draw=1;
		   break; 
		//Start Procedure-->User selects segments to keep
	   default: break;
    }
	

}

 void Setup()  
{ 
 
   //glEnable( GL_CULL_FACE );
 
   glShadeModel( GL_FLAT );
 
   glEnable(GL_DEPTH_TEST);
   glDepthFunc( GL_LEQUAL );      
   glClearDepth(1.0);             
 
 
   //Set up light source
   GLfloat ambientLight[] = { 0.2, 0.2, 0.2, 1.0 };
   GLfloat diffuseLight[] = { 0.8, 0.8, 0.8, 1.0 };
   GLfloat lightPos[] = { -20.0, 20.0, 150.0, 1.0 };
 
   glLightfv( GL_LIGHT0, GL_AMBIENT, ambientLight );
   glLightfv( GL_LIGHT0, GL_DIFFUSE, diffuseLight );
   glLightfv( GL_LIGHT0, GL_POSITION,lightPos );
  
   // polygon rendering mode and material properties
   glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
    
   glEnable(GL_COLOR_MATERIAL);
   glColorMaterial( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE );
    
   glEnable( GL_LIGHTING );
   glEnable( GL_LIGHT0);

   // Black background
   glClearColor(0.0f,0.0f,0.0f,1.0f);


   //Recorder-->Record PointCloud using Kinect
   //Uncomment in order to run recorder and capture scene
   cout<<"Run recorder"<<endl;
   clock_t startClock3 = clock();
   Recorder().run();
   clock_t endClock3 = clock();
   cout<<"Recording Time: "<< ((double)(endClock3 - startClock3)/CLOCKS_PER_SEC)<<" seconds"<<endl ;
   Loader().binarize();
   


    //Loader-->Load a PointCloud from a file
	cout<<"Run Loader"<<endl;
	points = Loader().load();
	
	
	cout<<"Run Normals Estimation Procedure"<<endl;
    normals_estimated=NormalsEstimation().run(points);

	if(points->size() != normals_estimated->size())
	{
		cout<<"Error cloud size is different from normals size"<<endl;
	}
	
	//Call CloudSegmentation Class to Start Segmentation procedure or Load Segmentation results from file
    CloudSegmentation().segment_Cloud( points,normals_estimated,indiciesAll);
	
}

 void MenuSelect(int choice)
{
     
    switch (choice) {
        case NORMALS : 
              if(enable_normals_draw!=0)
				enable_normals_draw=0;
		   else if(enable_normals_draw==0)
			    enable_normals_draw=1;
             break;
        case SEGMENTATION :
             if(enable_colored_cloud_draw!=0)
				enable_colored_cloud_draw=0;
		   else if(enable_colored_cloud_draw==0)
			    enable_colored_cloud_draw=1;
		   break; 
     
       
    }
    //
    glutPostRedisplay();
}

void Resize(int w, int h)
{ 
	// define the visible area of the window ( in pixels )
	if (h==0) h=1;
	glViewport(0,0,w,h); 

	// Setup viewing volume

	glMatrixMode(GL_PROJECTION); 
	glLoadIdentity();
	 
////(02b)
	          // L     R       B      T      N      F
	//glOrtho (-50.0f, 50.0f, -50.0f, 50.0f,-500.0f,500.0f);
	

	float aspect = (float)w/(float)h;             /// aspect ratio
	gluPerspective(60.0, aspect, 1.0, 500.0);
}

void Idle()
{
	glutPostRedisplay();
}

 int main (int argc,char* argv[])
 {
	 
   glutInit(&argc,argv);
   glutInitDisplayMode(GLUT_RGBA|GLUT_DEPTH|GLUT_DOUBLE);
   
   glutInitWindowSize(480,480);
   glutInitWindowPosition(50,50);
   glutCreateWindow("Test"); 
   Setup();
   
   glutDisplayFunc(Render);
   glutReshapeFunc(Resize);
   glutKeyboardFunc(Keyboard);
   glutMouseFunc(Mouse); 
   glutMotionFunc(MouseMotion);
   glutCreateMenu(MenuSelect);
	  glutAddMenuEntry("Normals Map",NORMALS);
	  glutAddMenuEntry("Segmentation",SEGMENTATION);
      // attach the menu to the right button
      glutAttachMenu(GLUT_RIGHT_BUTTON);
   glutSpecialFunc( process_F_Keys );
   glutIdleFunc(Idle);
   //Enter main event handling loop
   glutMainLoop();
	
   
   system("Pause");
   
   return 0;
 }