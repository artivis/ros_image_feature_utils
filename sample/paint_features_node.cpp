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

// ROS headers
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h>
#include <image_transport/subscriber_filter.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <dynamic_reconfigure/server.h>
#include <ros_image_feature_utils/paintConfig.h>

// OpenCV headers
#include <opencv2/highgui/highgui.hpp>

#include "ros_image_feature_utils/opencv_feature_extractor.h"

template <class T>
T getPrivateParam(const ros::NodeHandle& nh, const std::string& name, const T& default_val)
{
  T val;
  nh.param(name, val, default_val);
  ROS_DEBUG_STREAM_NAMED ("init", "Initialized " << name << " to " << val <<
                          "(default was " << default_val << ")");
  return val;
}

class ImageFeaturePainterNode
{
  typedef ros_image_feature_utils::paintConfig Reconf;
  typedef dynamic_reconfigure::Server<Reconf> ReconfServer;

public:

  ImageFeaturePainterNode() :
    _node("~"),
    _newimg(false),
    _imgnum(0),
    _it(_node)
  {
    std::string detector(getPrivateParam<std::string>(_node,   "feature_detector",   std::string("SURF")));
    std::string descriptor(getPrivateParam<std::string>(_node, "feature_descriptor", std::string("SURF")));

    _extractor.reset(new feature::FeatureExtractor(detector, descriptor));

    unsigned int keypts_lim(getPrivateParam<int>(_node, "keypoint_limit", 0));

    _extractor->limitKeypts(keypts_lim);

    _sub = _it.subscribe("image_in", 1, &ImageFeaturePainterNode::callback, this);

    _topic = _sub.getTopic();

    ROS_INFO_STREAM("Listening to : " << _topic);

    cv::namedWindow("view", cv::WINDOW_AUTOSIZE);
    cv::startWindowThread();

    _dsrv.reset(new ReconfServer);

    ReconfServer::CallbackType cb;
    cb = boost::bind(&ImageFeaturePainterNode::dynReconfCallback, this, _1, _2);

    _dsrv->setCallback(cb);
  }

  ~ImageFeaturePainterNode()
  {
    cv::destroyWindow("view");
  }

  bool process()
  {
    if (!_newimg)
      return false;

    boost::mutex::scoped_lock lock(_mut);

    std::string imagefilename = boost::str(boost::format( "%04d" ) % _imgnum )
                                  + "_" + _topic + ".jpg";

    feature::FeatureVector features = _extractor->process(_image.first);

    paintFeatures(features);

    _newimg = false;

    return (features.empty())? false : true;
  }

  void display()
  {
    boost::mutex::scoped_lock lock(_mut);

    cv::imshow("view", _image_painted);
    cv::waitKey(20);
  }

protected:

  bool _newimg;
  bool _paintimg;

  unsigned int _imgnum;

  std::string _topic;

  boost::mutex _mut;

  std::pair<cv::Mat, double> _image;
  cv::Mat _image_painted;

  ros::NodeHandle _node;

  ros_image_feature_utils::paintConfig _config;

  message_filters::Subscriber<sensor_msgs::Image> _subscriber;
  image_transport::ImageTransport _it;
  image_transport::Subscriber _sub;

  boost::shared_ptr<ReconfServer> _dsrv;

  boost::shared_ptr<feature::FeatureExtractor> _extractor;

  void callback(const sensor_msgs::ImageConstPtr& msgImg)
  {
    boost::mutex::scoped_lock lock(_mut);

    try
    {
      _image.first  = cv_bridge::toCvCopy(msgImg, "mono8")->image;
      _image.second = msgImg->header.stamp.toSec();
    }
    catch (cv_bridge::Exception)
    {
      ROS_ERROR("Couldn't convert %s image", msgImg->encoding.c_str());
      return;
    }

    _newimg = true;

    ++_imgnum;
  }

  void dynReconfCallback(ros_image_feature_utils::paintConfig &config, uint32_t level)
  {
    if (_config.feature_detector != config.feature_detector)
      _extractor->initDetector(config.feature_detector);

    if (_config.feature_descriptor != config.feature_descriptor)
      _extractor->initDescriptor(config.feature_descriptor);

    if (_config.keypoint_limit != config.keypoint_limit)
      _extractor->limitKeypts(config.keypoint_limit);

    _config = config;
  }

  void paintFeatures(const feature::FeatureVector& features)
  {
    std::vector<cv::KeyPoint> kpts;
    kpts.resize(features.size());

    for (size_t i=0; i<features.size(); ++i)
      kpts[i] = features[i];

    cv::drawKeypoints(_image.first, kpts, _image_painted, cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_feature_painter");

  ImageFeaturePainterNode painter;

  ros::Rate rate(25);

  while (ros::ok())
  {
    if (painter.process())
      painter.display();

    rate.sleep();

    ros::spinOnce();
  }

  return 0;
}
