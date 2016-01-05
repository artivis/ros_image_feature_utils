#ifndef OPENCV_FEATURE_EXTRACTOR_H
#define OPENCV_FEATURE_EXTRACTOR_H

#include "ros_image_feature_utils/opencv_feature.h"

#include <vector>
#include <opencv2/core/core.hpp>
#include <sensor_msgs/Image.h>
#include <boost/any.hpp>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/nonfree.hpp>

namespace feature
{
  /*
   * A class that wrap opencv feature detection and description
   * for an easy feature extraction
   */
  class FeatureExtractor
  {
    typedef std::vector<cv::KeyPoint>        KeyPointVector;

    typedef cv::Ptr<cv::FeatureDetector>     FeatureDetectorPtr;
    typedef cv::Ptr<cv::DescriptorExtractor> DescriptorExtractorPtr;

  public:

    FeatureExtractor();
    FeatureExtractor(const std::string& detector, const std::string& descriptor);

    ~FeatureExtractor() { }

    void init(const std::string& detector, const std::string& descriptor);

    FeatureVector process(const sensor_msgs::ImageConstPtr& sensor_image);
    FeatureVector process(const std::string& filename);
    FeatureVector process(const cv::Mat& image);

    std::string descriptorType();
    std::string detectorType();

    void descriptorParamList();
    void detectorParamList();

    bool initDetector(const std::string& detector_type);
    bool initDescriptor(const std::string& descriptor_type);

    bool limitKeypts(unsigned int nFeatures);

    int getKeyPointsLimit() { return this->_keypointsLimit; }

    bool isLimited() { return this->_isLimited; }

    bool isInit() { return this->_isInit; }

  private:

    bool _isInit;
    bool _isLimited;

    unsigned int  _keypointsLimit;

    FeatureDetectorPtr     _featureDetector;
    DescriptorExtractorPtr _featureDescriptor;

    KeyPointVector detectKeyPoints(const cv::Mat&);

    cv::Mat extractDescriptors(const cv::Mat&, KeyPointVector &);

    void listParam(cv::Algorithm* algorithm);
  };

} // namespace feature

#endif // OPENCV_FEATURE_EXTRACTOR_H
