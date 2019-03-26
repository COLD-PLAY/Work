// 1. 二维数组中的查找
public class Solution {
    public boolean Find(int target, int [][] array) {
        int height = array.length;
        int raw_height = height;
        int width = array[0].length;
        while (width > 0 && height > 0) {
            if (array[raw_height - height][width - 1] == target) return true;
            else if (array[raw_height - height][width - 1] > target) width -= 1;
            else height -= 1;
        }
        return false;
    }
}

// 2. 替换空格
public String replaceSpace(StringBuffer str) {
    StringBuilder string = new StringBuilder();
    for (int i = 0; i < str.length(); i++) {
        if (str.charAt(i) == ' ') string.append("%20");
        else string.append(str.charAt(i));
    }
    return string.toString();
}

// 3. 从头到尾打印链表
/**
*    public class ListNode {
*        int val;
*        ListNode next = null;
*
*        ListNode(int val) {
*            this.val = val;
*        }
*    }
*
*/
import java.util.ArrayList;
import java.util.Stack;
public class Solution {
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        ArrayList<Integer> array = new ArrayList<>();
        Stack<Integer> stack = new Stack<>();
        while (listNode != null) {
            stack.push(listNode.val);
            listNode = listNode.next;
        }
        while (!stack.empty()) {
            array.add(stack.pop());
        }
        return array;
    }
}

// 4. 重建二叉树
public class Solution {
    public TreeNode recursion(int [] pre, int pbegin, int pend, int [] in, int ibegin, int iend)
    {
        if (pbegin > pend || ibegin > iend) 
            return null;
        int pos = 0;
        for (pos = ibegin; pos < iend; pos++)
            if (in[pos] == pre[pbegin])
                break;
        TreeNode newNode = new TreeNode(pre[pbegin]);
        newNode.left = recursion(pre, pbegin + 1, pbegin + pos - ibegin, in, ibegin, pos - 1);
        newNode.right = recursion(pre, pbegin + pos - ibegin + 1, pend, in, pos + 1, iend);
        return newNode;
    }
    public TreeNode reConstructBinaryTree(int [] pre,int [] in)
    {
        return recursion(pre, 0, pre.length - 1, in, 0, in.length - 1);
    }
}

// 5. 用两个栈实现队列
import java.util.Stack;

public class Solution {
    Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();
    
    public void push(int node) {
        stack1.push(node);
    }
    
    public int pop() {
        int result;
        while (!stack1.empty()) {
            int temp = stack1.pop();
            stack2.push(temp);
        }
        result = stack2.pop();
        while (!stack2.empty()) {
            int temp = stack2.pop();
            stack1.push(temp);
        }
        return result;
    }
}

// 6. 旋转数组的最小数字
import java.util.ArrayList;
public class Solution {
    public int minNumberInRotateArray(int [] array) {
        int length = array.length;
        if (length == 0) return 0;
        for (int i = 0; i < length - 1; i++)
            if (array[i] > array[i + 1])
                return array[i + 1];
        return array[0];
    }
}

// 7. 斐波那契数列
public class Solution {
    public int Fibonacci(int n) {
        return Fibonacci(n, 0, 1);
    }
    private int Fibonacci(int n, int a, int b) {
        if (n == 0) return 0;
        if (n == 1) return b;
        return Fibonacci(n - 1, b, a + b);
    }
}

// 8. 跳台阶
public class Solution {
    public int JumpFloor(int target) {
        if (target == 1) return 1;
        if (target == 2) return 2;
        return JumpFloor(target - 1) + JumpFloor(target - 2);
    }
}

// 9. 变态跳台阶
public class Solution {
    public int JumpFloorII(int target) {
        if (target == 0) return 0;
        if (target == 1) return 1;
        int res = 0;
        for (int i = 0; i < target; i++) {
            res += JumpFloorII(i);
        }
        return res + 1;
    }
}

//  10. 矩形覆盖
public class Solution {
    public int RectCover(int target) {
        if (target <= 2) return target;
        int a = 1; int b = 2;
        for (int i = 3; i <= target; i++) {
            b = a + b;
            a = b - a;
        }
        return b;
    }
}

// 11. 二进制中1的个数
public class Solution {
    public int NumberOf1(int n) {
        int res = 0;
        while (n != 0) {
            n = (n - 1) & n;
            res += 1;
        }
        return res;
    }
}

// 12. 数值的整数次方
public class Solution {
    public double Power(double base, int exponent) {
        double res = 1;
        if (exponent > 0)
            while (exponent != 0) {
                res *= base; 
                exponent--;
            } 
        else if (exponent == 0);
        else
            while (exponent != 0) {
                res /= base;
                exponent++;
            }
        return res;
    }
}

// 13. 调整数组顺序使奇数位于偶数前面
public class Solution {
    public void reOrderArray(int [] array) {
        int len = array.length;
        boolean flag = true;
        for (int i = 0; i < len; i++) {
            flag = true;
            for (int j = 0; j < len - i - 1; j++) {
                if (array[j] % 2 == 0 && array[j + 1] % 2 == 1) {
                    int tem = array[j];
                    array[j] = array[j + 1];
                    array[j + 1] = tem;
                    flag = false;
                }
            }
            if (flag) break;
        }
    }
}

// 14. 链表中倒数第k个结点
/*
public class ListNode {
    int val;
    ListNode next = null;

    ListNode(int val) {
        this.val = val;
    }
}*/
import java.util.*;
public class Solution {
    public ListNode FindKthToTail(ListNode head,int k) {
        if (k <= 0) return null;
        ArrayList array = new ArrayList();
        ListNode p = head;
        while (p != null) {
            array.add(p);
            p = p.next;
        }
        if (k > array.size()) return null;
        for (int i = 0; i < array.size() - k; i++)
            head = head.next;
        return head;
    }
}

// 15. 反转链表
/*
public class ListNode {
    int val;
    ListNode next = null;

    ListNode(int val) {
        this.val = val;
    }
}*/
public class Solution {
    public ListNode ReverseList(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode p = head, q = p.next, s;
        p.next = null;
        while (q != null) {
            s = q.next;
            q.next = p;
            p = q; q = s;
        }
        return p;
    }
}

// 16. 合并两个排序的链表
/*
public class ListNode {
    int val;
    ListNode next = null;

    ListNode(int val) {
        this.val = val;
    }
}*/
public class Solution {
    public ListNode Merge(ListNode list1,ListNode list2) {
        ListNode list3 = new ListNode(-1);
        ListNode p = list3;
        while (list1 != null && list2 != null) {
            if (list1.val < list2.val) {
                ListNode node = new ListNode(list1.val);
                p.next = node; p = node; list1 = list1.next;
            }
            else {
                ListNode node = new ListNode(list2.val);
                p.next = node; p = node; list2 = list2.next;
            }
        }
        while (list1 != null) {
            ListNode node = new ListNode(list1.val);
            p.next = node; p = node; list1 = list1.next;
        }
        while (list2 != null) {
            ListNode node = new ListNode(list2.val);
            p.next = node; p = node; list2 = list2.next;
        }
        return list3.next;
    }
}

// 17. 树的子结构
/**
public class TreeNode {
    int val = 0;
    TreeNode left = null;
    TreeNode right = null;

    public TreeNode(int val) {
        this.val = val;
    }
}
*/
public class Solution {
    public boolean HasSubtree(TreeNode root1,TreeNode root2) {
        if (root1 == null || root2 == null) return false;
        return isSubtree(root1, root2) || HasSubtree(root1.left, root2) || HasSubtree(root1.right, root2);
    }
    public boolean isSubtree(TreeNode root1, TreeNode root2) {
        if (root2 == null) return true;
        if (root1 == null) return false;
        if (root1.val == root2.val)
            return isSubtree(root1.left, root2.left) && isSubtree(root1.right, root2.right);
        else return false;
    }
}

// 18. 二叉树的镜像
/**
public class TreeNode {
    int val = 0;
    TreeNode left = null;
    TreeNode right = null;

    public TreeNode(int val) {
        this.val = val;

    }

}
*/
public class Solution {
    public void Mirror(TreeNode root) {
        if (root == null) return;
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
        Mirror(root.left);
        Mirror(root.right);
    }
}

// 19. 顺时针打印矩阵

// 20. 包含min函数的栈
import java.util.Stack;

public class Solution {
    Stack<Integer> stack = new Stack<Integer>();
    Stack<Integer> min_stack = new Stack<Integer>();
    int min_ = 2147483647;
    public void push(int node) {
        if (node < min_) min_ = node;
        min_stack.push(min_);
        stack.push(node);
    }
    
    public void pop() {
        stack.pop();
        min_stack.pop();
    }
    
    public int top() {
        return stack.peek();
    }
    
    public int min() {
        return min_stack.peek();
    }
}

// 21. 栈的压入、弹出序列
import java.util.Stack;

public class Solution {
    public boolean IsPopOrder(int [] pushA,int [] popA) {
        Stack<Integer> stack = new Stack<Integer>();
        int i = 0, k = 0, len = pushA.length;
        for (; i < len; i++) {
            stack.push(pushA[i]);
            while (!stack.empty() && stack.peek() == popA[k]) {
                stack.pop();
                k++;
            }
        }
        if (k == len) return true;
        return false;
    }
}

// 22. 二叉搜索树的后序遍历序列
public class Solution {
    public boolean VerifySquenceOfBST(int [] sequence) {
        if (sequence.length == 0) return false;
        return IsBST(0, sequence.length - 1, sequence);
    }
    public boolean IsBST(int start, int end, int []sequence) {
        if (end <= start) return true;
        int i = start;
        for (; i < end; i++) {
            if (sequence[i] > sequence[end])
                break;
        }
        for (int j = i; j < end; j++) {
            if (sequence[j] < sequence[end])
                return false;
        }
        return IsBST(start, i - 1, sequence) && IsBST(i, end - 1, sequence);
    }
}

// 23. 二叉树中和为某一值的路劲
import java.util.ArrayList;
/**
public class TreeNode {
    int val = 0;
    TreeNode left = null;
    TreeNode right = null;

    public TreeNode(int val) {
        this.val = val;

    }

}
*/
public class Solution {
    private ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();
    private ArrayList<Integer> res_ = new ArrayList<Integer>();
    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root,int target) {
        if (root == null) return res;
        target -= root.val;
        res_.add(root.val);
        if (target == 0 && root.left == null && root.right == null) {
            res.add(new ArrayList<Integer>(res_));
        }
        FindPath(root.left, target);
        FindPath(root.right, target);
        res_.remove(res_.size() - 1);   // 回退
        return res;
    }
}

// 24. 复杂链表的复制
/*
public class RandomListNode {
    int label;
    RandomListNode next = null;
    RandomListNode random = null;

    RandomListNode(int label) {
        this.label = label;
    }
}
*/
public class Solution {
    public RandomListNode Clone(RandomListNode pHead)
    {
        RandomListNode res = new RandomListNode();
        if (pHead == null) return res;
        RandomListNode p = res;
        RandomListNode q = pHead;
        while (q) {
            
        }
    }
}
