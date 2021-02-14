#include <vector>
#include <stdlib.h>

//————————————————19/9/18———————————————————
// 1030. Matrix Cells in Distance Order
class Solution {
public:
	vector<vector<int>> allCellsDistOrder(int R, int C, int r0, int c0) {
		auto comp = [r0, c0](vector<int> &a, vector<int> &b) {
			return abs(a[0]-r0) + abs(a[1]-c0) < abs(b[0]-r0) + abs(b[1]-c0);
		};
		vector<vector<int>> res;
		for (int i = 0; i < R; i++) {
			for (int j = 0; j < C; j++) {
				vector<int> point = {i, j};
				res.push_back(point);
			}
		}
		sort(res.begin(), res.end(), comp);
		return res;
	}
};