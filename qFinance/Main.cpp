#include "Solver.h"
#include "Finance.h"

double angle_hour(int hour, int minute) {
	return 30. * ((hour % 12) + minute / 60.);
}

double angle_minute(int minute) {
	return 6. * minute;
}

struct HM {
	int hour;
	int minute;
	HM(int hour_, int minute_) : hour(hour_), minute(minute_) { }
};

void print(ostream& os, const HM& hm) {
	string s = to_string(hm.hour) + "/" + to_string(hm.minute);
	if (hm.hour < 10) s.insert(0, 1, '0');
	if (hm.minute < 10) s.insert(3, 1, '0');
	os << s << endl;
	cout << angle_hour(hm.hour, hm.minute) << " | " << angle_minute(hm.minute) << endl;
}

void hm_cross(const string& filename, int hour_end) {	
	ofstream out(filename);
	vector<HM> res;
	
	for (int hour = 0; hour <=hour_end; ++hour) {
		for (int minute = 0; minute <= 59; ++minute) {
			double cur = angle_minute(minute) - angle_hour(hour, minute);
			double next = angle_minute(minute + 1) - angle_hour(hour, minute + 1);
			if (cur <= 0. && next > 0.) {
				res.push_back(HM(hour, minute));
			}
		}
	}

	for (size_t i = 0; i < res.size(); ++i) {
		print(out, res[i]);
	}

}


int main() {
	hm_cross("hour_minute.dat", 23);	

	system("pause");
	return 0;
}