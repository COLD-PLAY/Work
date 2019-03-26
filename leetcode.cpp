// 2. 替换空格
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

// 3. 从头到尾打印链表
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

/**
 * Definition for binary tree
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */

// 4. 重建二叉树
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

// 5. 用两个栈实现队列
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

// 6. 旋转数组的最小数字
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

// 7. 斐波那契数列
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

// 8. 跳台阶
class Solution {
    public:
        int jumpFloor(int number) {
            if (number == 1) return 1;
            if (number == 2) return 2;
            return jumpFloor(number - 1) + jumpFloor(number - 2);
        }
};

// 9. 变态跳台阶
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

// 10. 矩形覆盖
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

// 11. 二进制中1的个数
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

// 12. 数值的整数次方
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