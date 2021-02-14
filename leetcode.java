// 1. Two Sum
class Solution {
    public int[] twoSum(int[] nums, int target) {
    	int length = nums.length;
        int[] diff = new int[length];
        int[] res = new int[2];
        int i = 0, j;
        for (int element: nums) {
        	diff[i++] = target - element;
        }
        for (i = 0; i < length; i++) {
        	for (j = i + 1; j < length; j++) {
        		if (diff[i] == nums[j]) {
        			res[0] = i; res[1] = j;
        			return res;
        		}        			
        	}
        }
        return res;
    }
}

// 2. Add Two Numbers
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode res = new ListNode(0);
        ListNode l3 = res;
        int d, c = 0, pre_c = 0;
        while (l1 != null || l2 != null) {
        	pre_c = c;
        	if (l1 != null && l2 != null) {
        		c = (l1.val + l2.val + pre_c) / 10;
        		d = (l1.val + l2.val + pre_c) % 10;
	        	l1 = l1.next;
	        	l2 = l2.next;
        	}
        	else if (l1 != null && l2 == null) {
        		c = (l1.val + pre_c) / 10;
        		d = (l1.val + pre_c) % 10;
	        	l1 = l1.next;
        	}
        	else {
        		c = (l2.val + pre_c) / 10;
        		d = (l2.val + pre_c) % 10;
	        	l2 = l2.next;
        	}
        	ListNode node = new ListNode(d);
        	node.next = null;
        	l3.next = node;
        	l3 = node;
        }
        if (c == 1) {
        	ListNode node = new ListNode(1);
        	node.next = null;
        	l3.next = node;
        }
        return res.next;
    }
}

// 3. Longest Substring Without Repeating Characters
class Solution {
    public int lengthOfLongestSubstring(String s) {
    	int res = 0;
    	String res_string = "";
    	int length = s.length();
       	for (int i = 0; i < length; i++) {
       		if (res_string.indexOf(s.charAt(i)) == -1) {
       			res_string += s.charAt(i);
       			if (res_string.length() > res) res = res_string.length();
       		}
       		else {
       			int pos = res_string.indexOf(s.charAt(i));
       			res_string = res_string.substring(pos + 1, res_string.length()) + String.valueOf(s.charAt(i));
       		}
       	}
       	return res;
    }
}

// 4. Median of Two Sorted Arrays
class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        
    }
}

// 27. Remove Element
class Solution {
    public int removeElement(int[] nums, int val) {
        int k = 0, length = nums.length;
        for (int i = 0; i < length; i++)
            if (nums[i] != val)
                nums[k++] = nums[i];
        return k;
    }
}