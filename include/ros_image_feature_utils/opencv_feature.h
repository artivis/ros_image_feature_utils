///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016, PAL Robotics S.L.
// All rights reserved.
//////////////////////////////////////////////////////////////////////////////
// Author: Jeremie Deray
//////////////////////////////////////////////////////////////////////////////

#ifndef OPENCV_FEATURE_H
#define OPENCV_FEATURE_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace
{
  const double CV_THRESH = 1e-9;
}

namespace feature
{

  const std::string FEATURE_EXTENSION = ".feat";

  typedef cv::Mat      Descriptor;
  typedef cv::KeyPoint Keypoint;

  class Feature : public Keypoint
  {
  public:

    Feature() : cv::KeyPoint() { }

    ~Feature() { }

    Feature(const cv::KeyPoint& kpts, const cv::Mat& desc) :
      cv::KeyPoint(kpts),
      descriptor(desc) { }

    Feature(const Feature& f) :
      cv::KeyPoint(f),
      descriptor(f.descriptor) { }

    Feature& operator=(const Feature& f)
    {
      angle      = f.angle;
      class_id   = f.class_id;
      octave     = f.octave;
      pt         = f.pt;
      response   = f.response;
      size       = f.size;
      descriptor = f.descriptor.clone();

      return *this;
    }

    bool operator==(const Feature& f)
    {
      if ( cv::norm(descriptor, f.descriptor) < CV_THRESH )
        if (pt == f.pt) // threshold ?
              return true;

      return false;
    }

    friend std::ostream &operator<<(std::ostream &out, const Feature &f)
    {
      out << std::endl;
      out << "keypoint : "    << std::endl;
      out << " angle : "      << f.angle       << std::endl;
      out << " class_id : "   << f.class_id    << std::endl;
      out << " octave : "     << f.octave      << std::endl;
      out << " pt : "         << f.pt          << std::endl;
      out << " response : "   << f.response    << std::endl;
      out << " size : "       << f.size        << std::endl;
      out << "descriptor : "  << f.descriptor  << std::endl;
      out << std::endl;

      return out;
    }

    void setPose(cv::Vec3d pose_in) {pose = pose_in;}

    Descriptor descriptor;

    cv::Vec3d pose;
  };

  typedef std::vector<Feature> FeatureVector;

  inline void operator += (FeatureVector& v1, const FeatureVector& v2)
  {
    v1.insert(v1.end(), v2.begin(), v2.end());
  }

//  class FeatureVector : public std::vector<Feature>
//  {
//  public:

//    FeatureVector() { }
//    ~FeatureVector() { }

//    FeatureVector& operator += (const FeatureVector& v2)
//    {
//      this->insert(end(), v2.begin(), v2.end());
//      return *this;
//    }

//    cv::Mat getFeatMat() const
//    {
//      cv::Mat desc = cv::Mat::zeros(size(),
//                                    this->at(0).descriptor.cols,
//                                    this->at(0).descriptor.type());

//      for (size_t i=0; i<size(); ++i)
//        this->at(i).descriptor.copyTo(desc.row(i));

//      return desc;
//    }
//  };

  void saveRichFeature(const std::string& filename, const Feature& feature);

  void loadRichFeature(const std::string& filename, Feature& feature);

  void saveRichFeature(const std::string& filename, const FeatureVector& features);

  void loadRichFeature(const std::string& filename, FeatureVector& features);

  void OpenCVToFeature(const std::vector<cv::KeyPoint>& kpts, const cv::Mat& desc,
                       FeatureVector& features);

  void FeatureToOpenCV(const FeatureVector& features,
                       std::vector<cv::KeyPoint>& kpts, cv::Mat& desc);

} //namespace feature

#endif // OPENCV_FEATURE_H
