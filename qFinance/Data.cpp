#include "Solver.h"

vector<double> read_rev_csv(const string& file_path) {
	// read a sequence of data from csv file into a vector, and reverse	
	vector<double> dat;
	ifstream in(file_path);
	string line;
	while (getline(in, line)) dat.push_back(stod(line));	
	reverse(dat.begin(), dat.end());
	return dat;
}

OnlineQuantile::OnlineQuantile(vector<double> dat, double p) 
: Np(5, 0.), dNp(5, 0.), N(5,0.) {
	assert(dat.size() == 5); // only 5 observations needed
	sort(dat.begin(), dat.end());
	Q = dat;
	for (int i = 0; i < 5; ++i) { N[i] = 1. + i; }
	Np[0]=1.; Np[1]=1.+2*p; Np[2]=1.+4*p; Np[3]=3.+2*p; Np[4]=5.;
	dNp[0]=0.; dNp[1]=p/2.; dNp[2]=p; dNp[3]=(1.+p)/2; dNp[4]=1.;	
}

void OnlineQuantile::addNumber(double dat) {
	int id = -1;
	if (dat < Q[0]) id = -1;
	else if (dat >= Q[0] && dat < Q[1]) id = 0;
	else if (dat >= Q[1] && dat < Q[2]) id = 1;
	else if (dat >= Q[2] && dat < Q[3]) id = 2;
	else if (dat >= Q[3] && dat < Q[4]) id = 3;
	else id = 4;

	if (id == -1) { Q[0] = dat; id = 0; }
	if (id == 4) { Q[4] = dat; id = 3; }
	for (int j = id + 1; j < 5; ++j) ++N[j];	
	for (int j = 0; j < 5; ++j) Np[j] += dNp[j];

	for (int j = 1; j <= 3; ++j) {
		double di = Np[j] - N[j];
		if ((di>=1.0 && N[j+1]-N[j]>1) || (di<=-1. && N[j-1]-N[j]<-1)) {
			di = (di > 0.0 ? 1.: -1.);
			double a = di / (N[j + 1] - N[j]);
			double b = N[j] - N[j - 1] + di;
			double c = N[j + 1] - N[j] - di;
			double d = (Q[j + 1] - Q[j]) / (N[j + 1] - N[j]);
			double e = (Q[j] - Q[j - 1]) / (N[j] - N[j - 1]);
			double qp = Q[j] + a*(b*d + c*e);
			if ((Q[j - 1] < qp) && (qp < Q[j + 1])) Q[j] = qp;
			else Q[j] += (Q[j+(int)di]-Q[j])*di / (N[j+(int)di]-N[j]);
			N[j] += di;
		}
	}
}

double OnlineQuantile::getValue() const {
	return Q[2];
}

OnlineQuantileB::OnlineQuantileB(vector<double> dat, double p) : B(dat.size()-1), p(p) {	
	sort(dat.begin(), dat.end());
	Q = dat;
	for (int i = 1; i <= B+1; ++i) { N.push_back(i); }
}

void OnlineQuantileB::addNumber(double dat) {
	int k = upper_bound(Q.begin(), Q.end(), dat) - Q.begin();	
	if (k == 0) { Q[0] = dat; k = 1; }
	if (k == B+1 && (Q[B] < dat)) { Q[B] = dat; k = B; }
	for (int j = k+1; j <= B+1; ++j) ++N[j-1]; // C++ convention N[j-1] is the jth	
	for (int j=2; j <= B; ++j) {
		double np = 1+(N[B]-1)*(j-1)/B;
		double d = np - N[j-1];
		if ((d>=1. && N[j]-N[j-1]>1.) || (d<=-1. && N[j-2]-N[j-1]<-1.)) {			
			d = (d>0. ? 1. : -1.);			
			double t1 = d/(N[j]-N[j-2]);			
			double t2 = N[j]-N[j-2]+d;
			double t3 = N[j]-N[j-2]-d;
			double t4 = (Q[j]-Q[j-1])/(N[j]-N[j-1]);
			double t5 = (Q[j-1]-Q[j-2])/(N[j-1]-N[j-2]);
			double q = Q[j-1] + t1*(t2*t4 + t3*t5);			
			if ((Q[j - 2] < q) && (q < Q[j])) { Q[j - 1] = q; }
			else {
				q = (Q[j-1+(int)d]-Q[j-1])*d/(N[j-1+(int)d]-N[j-1]);				
				Q[j - 1] += q;
			}
			N[j-1] += d;
		}		
	}	
}

double OnlineQuantileB::getValue() const { return Q[int(B*p)]; }

DataFrame::DataFrame(const string& file_path, char sep) {
	ifstream in(file_path);	
	string line;
	getline(in, line);
	
	istringstream is(line);
	string token;
	vector<string> cols;
	while (getline(is, token, sep)) cols.push_back(token);
	
	int ncol = cols.size();
	vector<vector<double> > data(ncol, vector<double>(0));
	while (getline(in, line)) {		
		istringstream is(line);		
		for (int i = 0; i < ncol; ++i) {
			getline(is, token, sep);
			data[i].push_back(stod(token));
		}
	}
	for (int i = 0; i < ncol; ++i) { mp[cols[i]] = data[i]; }
}


DataFrame::DataFrame(const vector<string>& keys, const vector<vector<double> >& dat) {
	if (keys.size() != dat.size()) 
		cout << "error: key size does not match with data size" << endl;
	int ncol = dat.size(), nrow = dat[0].size();
	for (int i = 0; i < ncol; ++i) { 
		mp[keys[i]] = dat[i];
		if (nrow != (int)(dat[i].size())) 
			cout << "error: dat size not equal" << endl;
	}
}

DataFrame::DataFrame(const DataFrame& df) : mp(df.mp) {}

DataFrame& DataFrame::operator=(const DataFrame& rhs) {
	if (&rhs == this) return *this;
	mp = rhs.mp;
	return *this;
}

vector<string> DataFrame::keys() const {
	vector<string> res;
	typedef map<string, vector<double>>::const_iterator MapIterator;	
	for (MapIterator iter = mp.begin(); iter != mp.end(); iter++)
		res.push_back(iter->first);
	return res;
}

vector<vector<double> > DataFrame::data() {
	vector<string> key_vec = keys();
	vector<vector<double> > res;
	for (size_t i = 0; i < key_vec.size(); ++i)
		res.push_back(mp[key_vec[i]]); 
	return res;
}

vector<double> DataFrame::getCol(const string& col) { return mp[col]; }

vector<int> DataFrame::dim() const {
	int ncol = mp.size();
	int nrow = mp.begin()->second.size();
	vector<int> res{ncol, nrow};
	return res;
}

void DataFrame::to_csv(const string& file_path) {	
	ofstream out(file_path);
	vector<string> key_vec = keys();	
	for (size_t i = 0; i < key_vec.size()-1; ++i) out << key_vec[i] << ",";
	out << key_vec.back() << endl;	
	vector<vector<double> > dat = data();
	int ncol = dat.size(), nrow = dat[0].size();
	for (int i = 0; i < nrow; ++i) {
		for (int j = 0; j < ncol - 1; ++j) out << dat[j][i] << ",";
		out << dat[ncol-1][i] << endl;
	}
}