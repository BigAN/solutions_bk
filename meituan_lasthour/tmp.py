class Solution(object):
    def findClosestLeaf(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        parents = {}
        leaves = []
        self.knode = None
        def traverse(root):
            if root.val == k: self.knode = root
            if not root.left and not root.right:
                leaves.append(root)
                return
            for child in (root.left, root.right):
                if not child: continue
                traverse(child)
                parents[child.val] = root
        def findParents(node):
            ans = [node.val]
            while node.val in parents:
                node = parents[node.val]
                ans.append(node.val)
            return ans
        traverse(root)
        kParents = findParents(self.knode)
        ans, dist = None, 0x7FFFFFFF
        for leaf in leaves:
            leafParents = findParents(leaf)
            cross = [n for n in leafParents if n in kParents][0]
            ndist = leafParents.index(cross) + kParents.index(cross)
            if ndist < dist:
                dist = ndist
                ans = leaf
        return ans.val

s = Solution()
s.findClosestLeaf([1,2,3,4,None,None,None,5,None,6],k=2)
