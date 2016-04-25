/*
 * Software License Agreement (Modified BSD License)
 *
 *  Copyright (c) 2016, PAL Robotics, S.L.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of PAL Robotics, S.L. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

/*
* Author: Jeremie Deray
*/

#include "ros_image_feature_utils/opencv_feature.h"

namespace feature
{

  void saveFeature(const std::string& filename, const Feature& feature)
  {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);

    if (!fs.isOpened())
      return;

    fs << "features " << "[";

    fs << "{" << "angle" << feature.angle << "classid" << feature.class_id << "octave" << feature.octave
       << "pose" << feature.pose << "pt" << feature.pt << "response" << feature.response << "size" << feature.size
       << "descriptor" << feature.descriptor << "}";

    fs << "]";

    fs.release();
  }

  void loadFeature(const std::string& filename, Feature& feature)
  {
    cv::FileStorage fs(filename, cv::FileStorage::READ);

    if (!fs.isOpened())
      return;

    cv::FileNode features   = fs["features"];
    cv::FileNodeIterator it = features.begin();

    (*it)["angle"]      >> feature.angle;
    (*it)["classid"]    >> feature.class_id;
    (*it)["octave"]     >> feature.octave;
    (*it)["pose"]       >> feature.pose;
    (*it)["pt"]         >> feature.pt;
    (*it)["response"]   >> feature.response;
    (*it)["size"]       >> feature.size;
    (*it)["descriptor"] >> feature.descriptor;

    fs.release();
  }

  void saveFeature(const std::string& filename, const FeatureVector& features)
  {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);

    if (!fs.isOpened())
      return;

    fs << "features " << "[";

    for (size_t i=0; i<features.size(); ++i)
    {
      fs << "{" << "angle" << features.at(i).angle << "classid" << features.at(i).class_id << "octave" << features.at(i).octave
         << "pose" << features.at(i).pose << "pt" << features.at(i).pt << "response" << features.at(i).response << "size" << features.at(i).size
         << "descriptor" << features.at(i).descriptor << "}";
    }

    fs << "]";

    fs.release();
  }

  void loadFeature(const std::string& filename, FeatureVector& features)
  {
    cv::FileStorage fs(filename, cv::FileStorage::READ);

    if (!fs.isOpened())
      return;

    features.clear();

    cv::FileNode features_file = fs["features"];
    cv::FileNodeIterator it = features_file.begin(), it_end = features_file.end();

    for ( ; it!=it_end; ++it)
    {
      Feature f;

      (*it)["angle"]      >> f.angle;
      (*it)["classid"]    >> f.class_id;
      (*it)["octave"]     >> f.octave;
      (*it)["pose"]       >> f.pose;
      (*it)["pt"]         >> f.pt;
      (*it)["response"]   >> f.response;
      (*it)["size"]       >> f.size;
      (*it)["descriptor"] >> f.descriptor;

      features.push_back(f);
    }

    fs.release();
  }

  void OpenCVToFeature(const std::vector<cv::KeyPoint>& kpts, const cv::Mat& desc,
                       FeatureVector& features)
  {
    if (!kpts.size() == desc.rows)
      return;

    features.clear();

    for (size_t i=0; i<kpts.size();++i)
      features.push_back( Feature(kpts[i], desc.row(i)) );
  }

  void FeatureToOpenCV(const FeatureVector& features,
                       std::vector<cv::KeyPoint>& kpts, cv::Mat& desc)
  {
    kpts.resize(features.size());
    desc = cv::Mat::zeros(features.size(), features[0].descriptor.rows, features[0].descriptor.type());

    for (size_t i=0; i<features.size(); ++i)
    {
      kpts[i]     = features[i];
      desc.row(i) = features[i].descriptor.clone();
    }
  }

} // namespace feature
