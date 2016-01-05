#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#include <boost/foreach.hpp>

#include "ros_image_feature_utils/opencv_feature_extractor.h"

namespace
{
  std::string uppercase(const std::string& s)
  {
    std::string u(s);
    std::transform(u.begin(), u.end(), u.begin(), ::toupper);

    return u;
  }
}

namespace feature
{

  FeatureExtractor::FeatureExtractor() :
    _isInit(false),
    _isLimited(false),
    _keypointsLimit(-1)
  {

  }

  FeatureExtractor::FeatureExtractor(const std::string &detector, const std::string &descriptor) :
    _isInit(false),
    _isLimited(false),
    _keypointsLimit(-1)
  {
    // TODO : check and filter
    // detector/descriptor combination
    init(detector, descriptor);

    if (!_isInit)
    {
      std::cerr << "Something went wrong!\nFeatureExtractor not initialized !" << std::endl;
    }
  }

  void FeatureExtractor::init(const std::string &detector, const std::string &descriptor)
  {
    if (!detector.compare("SIFT")   || !detector.compare("SURF") ||
        !descriptor.compare("SIFT") || !descriptor.compare("SURF"))
    {
      cv::initModule_nonfree();
    }

    bool det = initDetector(detector);
    bool des = initDescriptor(descriptor);

    _isInit = det && des;
  }

  FeatureVector FeatureExtractor::process(const sensor_msgs::ImageConstPtr &sensor_image)
  {
    cv_bridge::CvImagePtr image;
    try
    {
      image = cv_bridge::toCvCopy(sensor_image, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR_STREAM("cv_bridge exception: " << e.what());
      return FeatureVector();
    }

    return process( image->image );
  }

  FeatureVector FeatureExtractor::process(const std::string& filename)
  {
    if(!_isInit) return FeatureVector();

    ROS_DEBUG_STREAM("Processing file: " << filename);

    return process( cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE) );
  }

  FeatureVector FeatureExtractor::process(const cv::Mat& image)
  {
    if(!_isInit)
    {
      std::cout << "not init" << std::endl;
      return FeatureVector();
    }

    ROS_DEBUG_STREAM("Extracting features from " << image.rows
                     << " x " << image.cols
                     << " x " << image.channels()
                     << " image...");

    std::vector< cv::KeyPoint > keypoints = detectKeyPoints(image);

    ROS_DEBUG_STREAM("Keypoints Done!");

    cv::Mat features = extractDescriptors(image, keypoints);

    ROS_DEBUG_STREAM("Descriptors Done!");

    ROS_DEBUG_STREAM("Done! " << keypoints.size() << " features found.");

    FeatureVector feat_vec;

    OpenCVToFeature(keypoints, features, feat_vec);

    return feat_vec;
  }

  FeatureExtractor::KeyPointVector FeatureExtractor::detectKeyPoints(const cv::Mat& image)
  {
    std::vector< cv::KeyPoint > kpts;
    _featureDetector->detect(image, kpts);
    return kpts;
  }

  cv::Mat FeatureExtractor::extractDescriptors(const cv::Mat& image,
                                               KeyPointVector& kpts)
  {
    cv::Mat descriptors;
    _featureDescriptor->compute(image, kpts, descriptors);
    return descriptors;
  }

  bool FeatureExtractor::initDetector(const std::string& detector_type)
  {
    FeatureDetectorPtr tmp = cv::FeatureDetector::create(uppercase(detector_type));

    if (!tmp.empty())
    {
      _featureDetector = tmp;
      ROS_INFO_STREAM( "Feature detector type   : " << detectorType() );
    }
    else
    {
      ROS_ERROR_STREAM("Detector type " << detector_type << " doesn't exist !");
    }

    return !_featureDetector.empty();
  }

  bool FeatureExtractor::initDescriptor(const std::string& descriptor_type)
  {
    DescriptorExtractorPtr tmp = cv::DescriptorExtractor::create(uppercase(descriptor_type));

    if (!tmp.empty())
    {
      _featureDescriptor = tmp;
      ROS_INFO_STREAM( "Feature descriptor type : " << descriptorType() );
    }
    else
    {
      ROS_ERROR_STREAM("Descriptor type " << descriptor_type << " doesn't exist !");
    }

    return !_featureDescriptor.empty();
  }

  std::string FeatureExtractor::detectorType()
  {
    return (_featureDetector.empty()) ? std::string("Detector not Init") :
                                        _featureDetector->name();
  }

  std::string FeatureExtractor::descriptorType()
  {
    return (_featureDescriptor.empty()) ? std::string("Descriptor not Init") :
                                          _featureDescriptor->name();
  }

  bool FeatureExtractor::limitKeypts(unsigned int nFeatures)
  {
    if (!_isInit)
      return false;

    std::vector<std::string> parameters;
    _featureDetector->getParams(parameters);

    BOOST_FOREACH(const std::string &param, parameters)
    {
      if (param.compare("nFeatures") == 0)
      {
        int nf = (nFeatures > 0)? nFeatures : -1;
        _featureDetector->setInt("nFeatures", nf);
        _keypointsLimit = nf;

        return _isLimited = true;
      }
    }
  }

  void FeatureExtractor::listParam(cv::Algorithm* algorithm)
  {
    std::vector<std::string> parameters;
    algorithm->getParams(parameters);

    BOOST_FOREACH(const std::string &param, parameters)
    {
      int type = algorithm->paramType(param);
      std::string helpText = algorithm->paramHelp(param);
      std::string typeText;

      switch (type)
      {
      case cv::Param::BOOLEAN:
        typeText = "bool";
        break;
      case cv::Param::INT:
        typeText = "int";
        break;
      case cv::Param::REAL:
        typeText = "real (double)";
        break;
      case cv::Param::STRING:
        typeText = "string";
        break;
      case cv::Param::MAT:
        typeText = "Mat";
        break;
      case cv::Param::ALGORITHM:
        typeText = "Algorithm";
        break;
      case cv::Param::MAT_VECTOR:
        typeText = "Mat vector";
        break;
      }
      std::cout << "Parameter '" << param << "' type = " << typeText;

      if (!helpText.empty())
        std::cout << " help = " << helpText << std::endl;
      else
        std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;
  }

  void FeatureExtractor::detectorParamList()
  {
    std::string det = detectorType();

    if ( std::strcmp(det.c_str(), "Detector not Init") == 0)
    {
      std::cerr << det << std::endl;
      return;
    }

    std::cout << "************************************************************"   << std::endl;
    std::cout << "** Feature Points Detector " << det << " has the following parameters : **"   << std::endl;
    std::cout << "************************************************************\n" << std::endl;

    listParam(_featureDetector);
  }

  void FeatureExtractor::descriptorParamList()
  {
    std::string desc = descriptorType();

    if ( std::strcmp(desc.c_str(), "Descriptor not Init") == 0)
    {
      std::cerr << desc << std::endl;
      return;
    }

    std::cout << "**************************************************************"   << std::endl;
    std::cout << "** Feature Points Descriptor" << desc << " has the following parameters : **"   << std::endl;
    std::cout << "**************************************************************\n" << std::endl;

    listParam(_featureDescriptor);
  }

} // namespace feature
