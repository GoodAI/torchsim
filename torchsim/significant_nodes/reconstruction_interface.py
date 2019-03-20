from torchsim.core.graph.node_group import GroupInputs, GroupOutputs


class ClassificationInputs(GroupInputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.data = self.create("Data")
        self.label = self.create("Label")


class ClassificationOutputs(GroupOutputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.label = self.create("Label")
