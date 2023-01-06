import numpy as np

class Evaluator():
    def __init__(self):
        self.cmd1 = []
        self.cmd3 = []
        self.cmd5 = []
        self.add = []

    def cm_degree_n_metric(self, pose_pred, pose_target, n_max=0):
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_target[:, 3]) * 100
        rotation_diff = np.dot(pose_pred[:, :3], pose_target[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= n_max else n_max
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        return (translation_distance < n_max and angular_distance < n_max)
    
    def evaluate(self, pose_pred, pose_gt):
        if pose_pred is None:
            self.cmd5.append(False)
            self.cmd1.append(False)
            self.cmd3.append(False)
        else:
            if pose_pred.shape == (4, 4):
                pose_pred = pose_pred[:3, :4]
            if pose_gt.shape == (4, 4):
                pose_gt = pose_gt[:3, :4]
                
            metric_1 = self.cm_degree_n_metric(pose_pred, pose_gt, n_max=1)
            metric_3 = self.cm_degree_n_metric(pose_pred, pose_gt, n_max=3)
            metric_5 = self.cm_degree_n_metric(pose_pred, pose_gt, n_max=5)
            
            self.cmd1.append(metric_1)
            self.cmd3.append(metric_3)
            self.cmd5.append(metric_5)
            
    
    def summarize(self):
        cmd1 = np.mean(self.cmd1)
        cmd3 = np.mean(self.cmd3)
        cmd5 = np.mean(self.cmd5)
        print('1 cm 1 degree metric: {}'.format(cmd1))
        print('3 cm 3 degree metric: {}'.format(cmd3))
        print('5 cm 5 degree metric: {}'.format(cmd5))

        self.cmd1 = []
        self.cmd3 = []
        self.cmd5 = []
        return {'cmd1': cmd1, 'cmd3': cmd3, 'cmd5': cmd5}
