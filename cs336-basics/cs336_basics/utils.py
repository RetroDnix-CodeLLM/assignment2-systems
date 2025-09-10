class UnionFindSet:
    def __init__(self, size: int = 0):
        self.parent = [x for x in range(size)]
        self.size = [1] * size

    def find(self, x: int) -> int:
        # 路径压缩
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]

        return True

    def getSize(self, x: int) -> int:
        root = self.find(x)
        return self.size[root]
    
    def has(self, x: int) -> bool:
        return x >= 0 and x < len(self.parent)

def increaseD(d: dict, key: any, value: int = 1):
    d[key] = d.get(key, 0) + value

def decreaseD(d: dict, key: any, value: int = 1):
    if key in d:
        d[key] -= value
        if d[key] <= 0:
            del d[key]

def appendD(d: dict, key: any, value: any):
    if key not in d:
        d[key] = set()
    d[key].add(value)

def removeD(d: dict, key: any, value: any):
    if key in d and value in d[key]:
        d[key].remove(value)
