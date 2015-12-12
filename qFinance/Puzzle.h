#ifndef PUZZLE_H
#define PUZZLE_H

// input: hour, minute
// output: angle of the hour hand
double angle_hour(int hour, int minute);

// input: minute
// output: angle of the minute hand
double angle_minute(int minute);

// struct: hour, minute, to make the calculation convenient
struct HM {
	int hour;
	int minute;
	HM(int hour_, int minute_) : hour(hour_), minute(minute_) { }
};

// easy to output struct HM
void print(ostream& os, const HM& hm);

// used newton's method for calculation
// minute hand <= hour hange at minute && minute hand > hour hand at minute + 1 for each hour
void hm_cross(const string& filename, int hour_end);


// green book ,birthday problem in probability theory
//smallest number of people with same birthday has probability > 0.5
int birthday();


#endif