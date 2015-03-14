#! /usr/bin/env python

import rospy
import actionlib
from multiprocessing import Process, Queue
from human_trajectory.trajectory import Trajectory as Traj
from human_trajectory.msg import Trajectories
from human_movement_identifier.classifier import KNNClassifier
from human_movement_identifier.msg import HMCAction, HMCResult


class IdentifierServer(object):

    def __init__(self, name):
        self._action_name = name
        self.classifier = KNNClassifier()
        self.trajs = list()

        # Start server
        rospy.loginfo("%s is starting an action server", name)
        self._as = actionlib.SimpleActionServer(
            self._action_name,
            HMCAction,
            execute_cb=self.execute,
            auto_start=False
        )
        self._as.start()
        rospy.loginfo("%s is ready", name)

    # get trajectory data
    def traj_callback(self, msg):
        self.trajs = []
        for i in msg.trajectories:
            traj = Traj(i.uuid)
            traj.humrobpose = zip(i.trajectory, i.robot)
            traj.length.append(i.trajectory_length)
            traj.sequence_id = i.sequence_id
            self.trajs.append(traj)

    def get_online_prediction(self):
        # Subscribe to trajectory publisher
        rospy.loginfo(
            "%s is subscribing to human_trajectories/trajectories",
            self._action_name
        )
        s = rospy.Subscriber(
            "human_trajectories/trajectories", Trajectories,
            self.traj_callback, None, 30
        )

        while not self._as.is_preempt_requested():
            trajs = self.trajs
            for i in trajs:
                chunked_traj = self.classifier.create_chunk(
                    i.uuid, list(zip(*i.humrobpose)[0])
                )
                for j in chunked_traj:
                    self.classifier.predict_class_data(j)
                    if self._as.is_preempt_requested():
                        break
                if self._as.is_preempt_requested():
                    break
        self._as.set_preempted()
        s.unregister()

    # update classifier database
    def update_db(self):
        rospy.loginfo("%s is updating database", self._action_name)
        self.classifier.update_database()
        rospy.loginfo("%s is ready", self._action_name)

    # get the overal accuracy using 5-fold cross validation
    def get_accuracy(self):
        queue = Queue()
        t = Process(target=self.classifier.get_accuracy, args=(queue,))
        t.daemon = True
        t.start()
        preempt = False
        while t.is_alive():
            if self._as.is_preempt_requested():
                queue.put({'preempt':True})
                preempt = True
                break
            rospy.sleep(0.1)
        t.join()

        if not preempt:
            rospy.loginfo("The overal accuracy is %d",
                          self.classifier.accuracy)
            self._as.set_succeeded(HMCResult(False, self.classifier.accuracy))
        else:
            rospy.loginfo("The overall accuracy request is preempted")
            self._as.set_preempted()

    # execute call back for action server
    def execute(self, goal):
        if goal.request == 'update':
            self.update_db()
            self._as.set_succeeded(HMCResult(True, 0))
        elif goal.request == 'accuracy':
            self.get_accuracy()
        else:
            self.get_online_prediction()


if __name__ == '__main__':
    rospy.init_node("human_identifier_server")
    sv = IdentifierServer(rospy.get_name())
    rospy.spin()
