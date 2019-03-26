#### 1. 二维数组中的查找

```java
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
```
```python
class Solution:
# array 二维列表
    def Find(self, target, array):
        # write code here
        height = len(array)
        width = len(array[0])
        raw_height = height
        while (height > 0 and width > 0):
            if array[raw_height - height][width - 1] == target:
                return True
            elif array[raw_height - height][width - 1] > target:
                width -= 1
            else:
                height -= 1
        return False
```
#### 2. 替换空格

```c++
void replaceSpace(char *str,int length) {
    char string[length];
    for (int i = 0; i < length; i++)
        string[i] = str[i];
    for (int i = 0, k = 0; i < length; i++) {
        if (string[i] == ' ') {
            str[k++] = '%';
            str[k++] = '2';
            str[k++] = '0';
        }
        else str[k++] = string[i];
    }
}
```
```java
public String replaceSpace(StringBuffer str) {
	StringBuilder string = new StringBuilder();
    for (int i = 0; i < str.length(); i++) {
        if (str.charAt(i) == ' ') string.append("%20");
        else string.append(str.charAt(i));
    }
    return string.toString();
}
```
#### 3. 从头到尾打印链表

```c++
/**
*  struct ListNode {
*        int val;
*        struct ListNode *next;
*        ListNode(int x) :
*              val(x), next(NULL) {
*        }
*  };
*/
class Solution {
public:
    vector<int> printListFromTailToHead(ListNode* head) {
        vector<int> array;
        ListNode *p = head;
        while (p) {
            array.insert(array.begin(), p->val);
            p = p->next;
        }
        return array;
    }
};
```

```java
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
```

#### 4. 重建二叉树

```c++
/**
 * Definition for binary tree
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */

class Solution {
public:
    TreeNode* R(vector<int> a,int abegin,int aend,vector<int> b,int bbegin,int bend)
    {
        if(abegin>=aend || bbegin>=bend)
            return NULL;
        TreeNode* root=new TreeNode(a[abegin]);
        //root->val=a[abegin];
        int pivot;
        for(pivot=bbegin;pivot<bend;pivot++)
            if(b[pivot]==a[abegin])
                break;
        root->left=R(a,abegin+1,abegin+pivot-bbegin+1,b,bbegin,pivot);
        root->right=R(a,abegin+pivot-bbegin+1,aend,b,pivot+1,bend);
        return root;
    }
    TreeNode* reConstructBinaryTree(vector<int> pre,vector<int> vin) {
        return R(pre,0,pre.size(),vin,0,vin.size());
    }
};
```

```java
/**
 * Definition for binary tree
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
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
```

#### 5. 用两个栈实现队列

```c++
class Solution
{
public:
    void push(int node) {
        stack1.push(node);
    }

    int pop() {
        int result;
        while (!stack1.empty()) {
            int temp = stack1.top();
            stack1.pop();
            stack2.push(temp);
        }
        result = stack2.top();
        stack2.pop();
        while (!stack2.empty()) {
            int temp = stack2.top();
            stack2.pop();
            stack1.push(temp);
        }
        return result;
    }

private:
    stack<int> stack1;
    stack<int> stack2;
};
```

```java
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
```

#### 6. 旋转数组的最小数字

```c++
class Solution {
public:
    int minNumberInRotateArray(vector<int> rotateArray) {
        int length = rotateArray.size();
        if (length == 0) return 0;
        for (int i = 0; i < length - 1; i++)
            if (rotateArray[i] > rotateArray[i + 1])
                return rotateArray[i + 1];
        return rotateArray[0];
    }
};
```
```java
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
```

#### 7. 斐波那契数列

```java
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
```

```c++
class Solution {
public:
    int Fibonacci(int n) {
        return F(n, 0, 1);
    }
    int F(int n, int a, int b) {
        if (n == 0) return 0;
        if (n == 1) return b;
        return F(n - 1, b, a + b);
    }
};
```

#### 8. 跳台阶

```c++
class Solution {
public:
    int jumpFloor(int number) {
        if (number == 1) return 1;
        if (number == 2) return 2;
        return jumpFloor(number - 1) + jumpFloor(number - 2);
    }
};
```

```java
public class Solution {
    public int JumpFloor(int target) {
        if (target == 1) return 1;
        if (target == 2) return 2;
        return JumpFloor(target - 1) + JumpFloor(target - 2);
    }
}
```

#### 9. 变态跳台阶

```c++
class Solution {
public:
    int jumpFloorII(int number) {
        if (number == 0) return 0;
        if (number == 1) return 1;
        int res;
        for (int i = 0; i < number; i++)
            res += JumpFloorII(i);
        return res + 1;
    }
};
```



```java
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
```

#### 10. 矩形覆盖

```c++
class Solution {
public:
    int rectCover(int number) {
        if (number <= 2) return number;
        vector<int> array;
        array.push_back(0);
        array.push_back(1);
        array.push_back(2);
        for (int i = 3; i <= number; i++)
            array.push_back(array[i - 1] + array[i - 2]);
        return array[number];
    }
};
```

```java
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
```

#### 11. 二进制中1的个数

```c++
class Solution {
public:
     int  NumberOf1(int n) {
         int res = 0;
         while (n != 0) {
             n = n & (n - 1);
             res++;
         }
         return res;
     }
};
```

```java
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
```

#### 12. 数值的整数次方

```java
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
```

```c++
class Solution {
public:
    double Power(double base, int exponent) {
        double res = 1;
        if (exponent > 0) while (exponent != 0) {
            res *= base;
            exponent--;
        }
        else if (exponent < 0) while (exponent != 0) {
            res /= base;
            exponent++;
        }
        return res;
    }
};
```

#### 13. 调整数组顺序使奇数位于偶数前面

```java
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
```

```c++
class Solution {
public:
    void reOrderArray(vector<int> &array) {
        int flag = 1, len = array.size();
        for (int i = 0; i < len; i++) {
            flag = 1;
            for (int j = 0; j < len - i - 1; j++)
                if (array[j] % 2 == 0 && array[j + 1] % 2 == 1) {
                    int tem = array[j];
                    array[j] = array[j + 1];
                    array[j + 1] = tem;
                    flag = 0;
                }
            if (flag) break;
        }
    }
};
```

#### 14. 链表中倒数第k个结点

```c++
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* FindKthToTail(ListNode* pListHead, unsigned int k) {
        if (pListHead == NULL) return NULL;
        vector<ListNode*> array;
        while (pListHead != NULL) {
            array.push_back(pListHead);
            pListHead = pListHead->next;
        }
        if (k > array.size()) return NULL;
        return array[array.size() - k];
    }
};
```

```java
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
```

#### 15. 反转链表

```java
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
```

```c++
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* ReverseList(ListNode* pHead) {
        if (pHead == NULL || pHead->next == NULL) return pHead;
        ListNode *p = pHead, *q = pHead->next, *s;
        p->next = NULL;
        while (q) {
            s = q->next;
            q->next = p;
            p = q; q = s;
        }
        return p;
    }
};
```

#### 16. 合并两个排序的链表

```c++
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* Merge(ListNode* pHead1, ListNode* pHead2)
    {
        ListNode *pHead3 = new ListNode(-1);
        ListNode *p3 = pHead3;
        while (pHead1 && pHead2) {
            if (pHead1->val < pHead2->val) {
                ListNode *node = new ListNode(pHead1->val);
                p3->next = node; p3 = node; pHead1 = pHead1->next;
            }
            else {
                ListNode *node = new ListNode(pHead2->val);
                p3->next = node; p3 = node; pHead2 = pHead2->next;
            }
        }
        while (pHead1) {
            ListNode *node = new ListNode(pHead1->val);
            p3->next = node; p3 = node; pHead1 = pHead1->next;
        }
        while (pHead2) {
            ListNode *node = new ListNode(pHead2->val);
            p3->next = node; p3 = node; pHead2 = pHead2->next;
        }
        return pHead3->next;
    }
};
```

```java
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
```

#### 17. 树的子结构

```java
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
```

```c++
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
    bool isSubtree(TreeNode* pRoot1, TreeNode* pRoot2) {
        if (pRoot2 == NULL) return true;
        if (pRoot1 == NULL) return false;
        if (pRoot1->val == pRoot2->val)
            return isSubtree(pRoot1->left, pRoot2->left) && isSubtree(pRoot1->right, pRoot2->right);
        else return false;
    }
public:
    bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2)
    {
        if (pRoot1 == NULL || pRoot2 == NULL) return false;
        return isSubtree(pRoot1, pRoot2) || HasSubtree(pRoot1->left, pRoot2) || HasSubtree(pRoot1->right, pRoot2);
    }
};
```

#### 18. 二叉树的镜像

```java
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
```

```c++
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    void Mirror(TreeNode *pRoot) {
        if (pRoot == NULL) return;
        TreeNode *temp = pRoot->left;
        pRoot->left = pRoot->right;
        pRoot->right = temp;
        Mirror(pRoot->left);
        Mirror(pRoot->right);
    }
};
```

#### 19. 顺时针打印矩阵

```c++
class Solution {
public:
    vector<int> printMatrix(vector<vector<int>> matrix) {
        vector<int> res;
        while (matrix.size() >= 2 && matrix[0].size() >= 2) {
            for (int i = 0; i < matrix[0].size(); i++) res.push_back(matrix[0][i]);
            matrix.erase(matrix.begin());
            for (int i = 0; i < matrix.size(); i++) {
                res.push_back(matrix[i][matrix[i].size() - 1]);
                matrix[i].pop_back();
            }
            for (int i = matrix[0].size() - 1; i >= 0; i--) res.push_back(matrix[matrix.size() - 1][i]);
            matrix.erase(matrix.end());
            for (int i = matrix.size() - 1; i >= 0; i--) {
                res.push_back(matrix[i][0]);
                matrix[i].erase(matrix[i].begin());
            }
        }
        if (matrix[0].size() == 0 || matrix.size() == 0);
        else if (matrix.size() == 1) {
            for (int i = 0; i < matrix[0].size(); i++) res.push_back(matrix[0][i]);
        }
        else {
            for (int i = 0; i < matrix.size(); i++) res.push_back(matrix[i][0]);
        }
        return res;
    }
};
```

#### 20. 包含min函数的栈

```java
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
```

```c++
class Solution {
public:
    void push(int value) {
        if (value < min_) min_ = value;
        stackMin.push(min_);
        stackInt.push(value);
    }
    void pop() {
        stackInt.pop();
        stackMin.pop();
    }
    int top() {
        return stackInt.top();
    }
    int min() {
        return stackMin.top();
    }
private:
    stack<int> stackInt;
    stack<int> stackMin;
    int min_ = 2147483647;
};
```

#### 21. 栈的压入、弹出序列

``````java
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
``````

```c++
class Solution {
public:
    bool IsPopOrder(vector<int> pushV,vector<int> popV) {
        stack<int> s;
        int i = 0, j = 0, len = pushV.size();
        for (; i < len; i++) {
            s.push(pushV[i]);
            while (!s.empty() && s.top() == popV[j]) {
                s.pop();
                j++;
            }
            if (j == len) return true;
            return false;
        }
    }
};
```

#### 22. 二叉搜索树的后序遍历序列
```java

```