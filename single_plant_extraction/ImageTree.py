import numpy as np
class IDBoxTree:
    def __init__(self, id, id_center,goal_center, direction, idbox, goalbox):
        """
        初始化IDBoxTree类的实例。
        :param id: int, 标识符
        :param tree_center: list, 包含x, y, z坐标的数组
        :param direction: str, 方向,"0代表左边，1代表右边"
        :param box: dict, 包含top, bottom, left, right的数组
        """
        self.id = id
        self.id_center = id_center
        self.goal_center = goal_center
        self.direction = direction
        self.idbox = idbox
        self.goalbox=goalbox

class GoalTree:
    def __init__(self,goal_box,goal_box_match):
        self.goal_box = goal_box
        self.goal_box_match =goal_box_match   
class ImageTree:
    def __init__(self, image_name):
        """
        初始化ImageTree类的实例。

        :param image_name: str, 相机的名字
        """
        self.image_name = image_name
        self.pr_tree = []  # 这个列表将包含IDBoxTree类的实例
        self.goal_tree = []
    def add_id_box_tree(self, id_box_tree):
        """
        向pr_tree列表中添加一个新的IDBoxTree实例。

        :param id_box_tree: IDBoxTree, 要添加的IDBoxTree实例
        """
        if isinstance(id_box_tree, IDBoxTree):
            self.pr_tree.append(id_box_tree)
        else:
            raise ValueError("Only instances of IDBoxTree can be added to the pr_tree list.")
 
    def add_goal_tree(self, goal_tree):
        """
        向goal_tree列表中添加一个新的GoalTree实例。

        :param goal_tree: GoalTree, 要添加的GoalTree实例
        """
        if isinstance(goal_tree, GoalTree):
            self.goal_tree.append(goal_tree)
        else:
            raise ValueError("Only instances of IDBoxTree can be added to the pr_tree list.")
    
    def update_goal_box_match(self, goal_box):
        """
        如果goal_tree列表中存在指定的goal_box，将其对应的goal_box_match置为1。

        :param goal_box: 与目标GoalTree实例的goal_box进行匹配的值
        """

        for i,gt in enumerate(self.goal_tree):
            if np.array_equal(gt.goal_box,goal_box):
                self.goal_tree[i].goal_box_match = 1
                break

    def check_id_in_pr_tree(self, id_number):
        """
        检查给定的ID号码是否在pr_tree列表中存在，并判断对应的goalbox是否存在。

        :param id_number: int, 要检查的ID号码
        :return: int, id不存在为0，id存在，box不存在为1，id存在 box存在为2
        """
        for goal in self.pr_tree:
            if goal.id == id_number:  # 假设pr有一个id属性
                if goal.goalbox is not None:  # 假设pr有一个goalbox属性
                    return 2
                else:
                    return 1
        return 0

    def update_goal_box_in_pr_tree(self, id_number, new_box):
        """
        根据输入的ID更新pr_tree中对应的goal_box属性。

        :param id_number: int, 要更新的ID号码
        :param new_box: 任何类型, 用于更新goal_box的数据
        :return: bool, 如果更新成功返回True，否则返回False
        """
        for id_box in self.pr_tree:
            if id_box.id == id_number:  # 假设IDBoxTree有一个id属性
                id_box.goalbox = new_box  # 更新goal_box属性
                return True
        return False      

class IDSET:
    def __init__(self,id,id_center_3d,goal_center_3d,id_num,goal_num):
        self.id=id
        self.id_center_3d=id_center_3d
        self.id_num=id_num
        self.goal_center_3d=goal_center_3d
        self.goal_num=goal_num
    def update_id_center_3d(self, new_id_center_3d):
        # 直接更新 id_center_3d
        self.id_center_3d = new_id_center_3d

    def update_goal_center_3d(self, new_goal_center_3d):
        # 直接更新 goal_center_3d
        self.goal_center_3d = new_goal_center_3d    
    def increment_id_num(self):
        # id_num 加 1
        self.id_num += 1

    def increment_goal_num(self):
        # goal_num 加 1
        self.goal_num += 1    



# 使用示例
if __name__ == "__main__":
    # 创建IDBoxTree实例
    id_box_tree_example = IDBoxTree(
        id=1,
        id_center=[100, 150, 0],
        goal_center=[200, 250, 0],
        direction="1",  # 表示右边
        idbox={"top": 100, "bottom": 50, "left": 90, "right": 110},
        goalbox={"top": 200, "bottom": 150, "left": 190, "right": 210}
    )

    # 创建ImageTree实例
    image_tree_example = ImageTree(image_name="Camera1")

    # 将IDBoxTree实例添加到ImageTree的pr_tree列表中
    image_tree_example.add_id_box_tree(id_box_tree_example)
    image_tree_example.update_goal_box_in_pr_tree(1,{"top": 2, "bottom": 1, "left": 1, "right": 2})
    # 打印信息以验证
    print(f"Camera Name: {image_tree_example.image_name}")
    for tree in image_tree_example.pr_tree:
        print(f"ID: {tree.id}, ID Center: {tree.id_center}, Goal Center: {tree.goal_center}, Direction: {tree.direction}")
        print(f"ID Box: {tree.idbox}, Goal Box: {tree.goalbox}")