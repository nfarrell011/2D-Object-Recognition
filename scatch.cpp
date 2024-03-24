    float minProjE1 = std::numeric_limits<float>::max();
    float maxProjE1 = std::numeric_limits<float>::lowest();
    float minProjE2 = std::numeric_limits<float>::max();
    float maxProjE2 = std::numeric_limits<float>::lowest();

    // Project points onto e1 and e2
    for (const auto& point : points) {
        float projE1 = point.dot(e1);
        float projE2 = point.dot(e2);

        minProjE1 = std::min(minProjE1, projE1);
        maxProjE1 = std::max(maxProjE1, projE1);
        minProjE2 = std::min(minProjE2, projE2);
        maxProjE2 = std::max(maxProjE2, projE2);
    }

    std::vector<cv::Point2f> corners;
    corners.push_back(cv::Point2f(minProjE1, minProjE2));
    corners.push_back(cv::Point2f(maxProjE1, minProjE2));
    corners.push_back(cv::Point2f(maxProjE1, maxProjE2));
    corners.push_back(cv::Point2f(minProjE1, maxProjE2));

    for (int i = 0; i < corners.size(); i++) {
        cv::line(src, corners[i], corners[(i + 1) % corners.size()], cv::Scalar(0, 255, 0), 2);
}