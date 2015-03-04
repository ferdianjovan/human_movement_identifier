#!/usr/bin/env python

import sys
import random
import rospy
import math
import pymongo
import pylab
import matplotlib.pyplot as plt
from collections import namedtuple
from mpl_toolkits.axes_grid.axislines import SubplotZero
from human_trajectory.trajectory import Trajectory
from geometry_msgs.msg import Point, Quaternion, Pose, PoseStamped
from std_msgs.msg import Header


class KNNClassifier(object):

    def __init__(self, training_ratio, chunk, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.k = 11
        self.accuracy = 0
        self.training_data = []
        self.test_data = []
        self.LabeledNormalizedPoses = namedtuple(
            "NormalizePoses", "uuid real normal"
        )

        trajs = self._retrieve_logs()
        self._split_and_label_data(trajs, training_ratio, chunk)

    # get k nearest values to a test data based on positions and velocities
    def _nearest_values_to(self, test):
        index = []
        nearest = []
        test_poses = test.normal
        for i, j in enumerate(self.training_data):
            dist = 0
            vel = 0
            for k, l in enumerate(j[0].normal):
                # delta distance calculation
                dx = test_poses[k].pose.position.x - l.pose.position.x
                dy = test_poses[k].pose.position.y - l.pose.position.y
                dist += math.hypot(dx, dy)
                # delta velocity calculation
                if k >= 1:
                    dx = l.pose.position.x - j[0].normal[k-1].pose.position.x
                    dy = l.pose.position.y - j[0].normal[k-1].pose.position.y
                    velo_l = math.hypot(dx, dy) / (
                        (l.header.stamp.secs -
                         j[0].normal[k-1].header.stamp.secs) +
                        (l.header.stamp.nsecs -
                         j[0].normal[k-1].header.stamp.nsecs) /
                        math.pow(10, 9)
                    )
                    dx = test_poses[k].pose.position.x - \
                        test_poses[k-1].pose.position.x
                    dy = test_poses[k].pose.position.y - \
                        test_poses[k-1].pose.position.y
                    velo_test = math.hypot(dx, dy) / (
                        (test_poses[k].header.stamp.secs -
                         test_poses[k-1].header.stamp.secs) +
                        (test_poses[k].header.stamp.nsecs -
                         test_poses[k-1].header.stamp.nsecs) /
                        math.pow(10, 9)
                    )
                    vel += abs(velo_l - velo_test)
            if nearest != []:
                dist = (self.alpha * dist) + (self.beta * vel)
                max_val = max(nearest)
                if max_val > dist and len(nearest) >= self.k:
                    temp = nearest.index(max_val)
                    nearest[temp] = dist
                    index[temp] = i
                elif max_val > dist and len(nearest) < self.k:
                    nearest.append(dist)
                    index.append(i)
            else:
                nearest.append(dist)
                index.append(i)

        return [self.training_data[i] for i in index]

    # predict the class of the test data
    def predict_class_data(self, test_data):
        rospy.loginfo("Predicting class for %s", test_data.uuid)
        result = None
        nn = self._nearest_values_to(test_data)
        human = [i for i in nn if i[1] == 'human']
        nonhuman = [i for i in nn if i[1] == 'non-human']
        rospy.loginfo("Vote: %d, %d", len(human), len(nonhuman))
        if len(human) > len(nonhuman):
            result = 'human'
        else:
            result = 'non-human'

        rospy.loginfo("%s belongs to %s", test_data.uuid, result)
        return (result, human[:3], nonhuman[:3])

    # get accuracy of the overall prediction
    def get_accuracy(self):
        rospy.loginfo("Getting the overall accuracy...")
        counter = 0
        for i in self.test_data:
            result = self.predict_class_data(i[0])
            rospy.loginfo("The actual class is %s", i[1])
            if result[0] == i[1]:
                counter += 1
                print float(counter) / float(len(self.test_data))
        self.accuracy = float(counter) / float(len(self.test_data))

        return self.accuracy

    # split and label data into training and test set
    def _split_and_label_data(self, trajs, training_ratio, chunk):
        rospy.loginfo("Splitting data...")
        for uuid, traj in trajs.iteritems():
            traj.validate_all_poses()
            chunked_traj = self._create_chunk(
                uuid, list(zip(*traj.humrobpose)[0]), chunk
            )
            label = 'human'
            start = traj.humrobpose[0][0].header.stamp
            end = traj.humrobpose[-1][0].header.stamp
            if traj.length[-1] < 0.1 or (end - start).secs < 3:
                label = 'non-human'
            for i in chunked_traj:
                if random.random() < training_ratio:
                    self.training_data.append((i, label))
                else:
                    self.test_data.append((i, label))

    # normalize poses so that the first pose becomes (0,0)
    # and the second pose becomes the base for the axis
    def get_normalized_poses(self, poses):
        dx = abs(poses[1].pose.position.x - poses[0].pose.position.x)
        dy = abs(poses[1].pose.position.y - poses[0].pose.position.y)
        if dx < 0.00001:
            dx = 0.00000000000000000001
        rad = math.atan(dy / dx)
        rot_matrix = [
            [math.cos(rad), -math.sin(rad)],
            [math.sin(rad), math.cos(rad)]
        ]
        for i, j in enumerate(poses):
            x = j.pose.position.x * rot_matrix[0][0] + \
                j.pose.position.y * rot_matrix[1][0]
            y = j.pose.position.x * rot_matrix[0][1] + \
                j.pose.position.y * rot_matrix[1][1]
            poses[i].pose.position.x = x
            poses[i].pose.position.y = y
            if i != 0:
                poses[i].pose.position.x -= poses[0].pose.position.x
                poses[i].pose.position.y -= poses[0].pose.position.y

        poses[0].pose.position.x = poses[0].pose.position.y = 0
        return poses

    # chunk data for each trajectory
    def _create_chunk(self, uuid, poses, chunk):
        i = 0
        chunk_trajectory = []
        while i < len(poses) - (chunk - 1):
            normalized = list()
            # can not just do poses[i:i+chunk], need to rewrite
            for j in range(chunk):
                position = Point(
                    poses[i + j].pose.position.x,
                    poses[i + j].pose.position.y,
                    poses[i + j].pose.position.z
                )
                orientation = Quaternion(
                    poses[i + j].pose.orientation.x,
                    poses[i + j].pose.orientation.y,
                    poses[i + j].pose.orientation.z,
                    poses[i + j].pose.orientation.w
                )
                pose = Pose(position, orientation)
                header = Header(
                    poses[i + j].header.seq,
                    poses[i + j].header.stamp,
                    poses[i + j].header.frame_id
                )
                normalized.append(PoseStamped(header, pose))
            normalized = self.get_normalized_poses(normalized)
            chunk_trajectory.append(
                self.LabeledNormalizedPoses(uuid, poses[i:i+chunk], normalized)
            )
            i += chunk

        return chunk_trajectory

    # retrieve trajectory from mongodb
    def _retrieve_logs(self):
        client = pymongo.MongoClient(
            rospy.get_param("datacentre_host", "localhost"),
            rospy.get_param("datacentre_port", 62345)
        )
        rospy.loginfo("Retrieving data from mongodb...")
        trajs = dict()
        rospy.loginfo("Constructing data from people perception...")
        for log in client.message_store.people_perception.find():
            for i, uuid in enumerate(log['uuids']):
                if uuid not in trajs:
                    t = Trajectory(uuid)
                else:
                    t = trajs[uuid]
                t.append_pose(log['people'][i],
                              log['header']['stamp']['secs'],
                              log['header']['stamp']['nsecs'],
                              log['robot'])
                trajs.update({uuid: t})
        return trajs

    # create a visualisation graph in cartesian coordinate
    def visualize_test_between_class(self, test, human, non_human):
        fig = plt.figure("Trajectories for Test, Human, and Non-Human")
        ax = SubplotZero(fig, 111)
        fig.add_subplot(ax)
        line_style = ['r.-', 'gx-', 'bo-']
        avg = 0

        # plotting test data
        x = [i.pose.position.x for i in test]
        y = [i.pose.position.y for i in test]
        ax.plot(x, y, line_style[0], label="Test")
        avg += sum([abs(i) for i in x]) / float(len(x))
        avg += sum([abs(i) for i in y]) / float(len(y))
        # plotting human data
        x = [i.pose.position.x for i in human]
        y = [i.pose.position.y for i in human]
        ax.plot(x, y, line_style[1], label="Human")
        avg += sum([abs(i) for i in x]) / float(len(x))
        avg += sum([abs(i) for i in y]) / float(len(y))
        # plotting non-human data
        x = [i.pose.position.x for i in non_human]
        y = [i.pose.position.y for i in non_human]
        ax.plot(x, y, line_style[2], label="Non-human")
        avg += sum([abs(i) for i in x]) / float(len(x))
        avg += sum([abs(i) for i in y]) / float(len(y))
        avg /= 60.0

        ax.margins(0.05)
        ax.legend(loc="lower right", fontsize=10)
        plt.title("Chunks of Trajectories")
        plt.xlabel("Axis")
        plt.ylabel("Ordinate")

        for direction in ["xzero", "yzero"]:
            ax.axis[direction].set_axisline_style("-|>")
            ax.axis[direction].set_visible(True)

        for direction in ["left", "right", "bottom", "top"]:
            ax.axis[direction].set_visible(False)

        pylab.grid()
        plt.show()

if __name__ == '__main__':
    rospy.init_node("labeled_short_poses")

    if len(sys.argv) < 6:
        rospy.logerr(
            "usage: predictor train_ratio chunk alpha beta accuracyOrNot[1/0]"
        )
        sys.exit(2)

    lsp = KNNClassifier(
        float(sys.argv[1]), int(sys.argv[2]),
        float(sys.argv[3]), float(sys.argv[4]))
    if int(sys.argv[5]):
        rospy.loginfo("The overall accuracy is " + str(lsp.get_accuracy()))
    else:
        human_data = None
        while not rospy.is_shutdown():
            human_data = lsp.test_data[random.randint(0, len(lsp.test_data)-1)]
            # if human_data[1] == 'non-human':
            prediction = lsp.predict_class_data(human_data[0])
            rospy.loginfo("The actual class is %s", human_data[1])
            for i in range(min([len(prediction[1]), len(prediction[2])])):
                rospy.loginfo("The real data visualisation")
                lsp.visualize_test_between_class(
                    human_data[0].real,
                    prediction[1][i][0].real,
                    prediction[2][i][0].real
                )
                rospy.loginfo("The normalized data visualisation")
                lsp.visualize_test_between_class(
                    human_data[0].normal,
                    prediction[1][i][0].normal,
                    prediction[2][i][0].normal
                )
