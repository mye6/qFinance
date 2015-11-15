#include "Solver.h"

/*
vector<double> dat{ 1, 2, 3};
nrow = -1 (default)
output vector:
1
2
3
*/
VectorXd vecGen(vector<double> dat, int nrow) {	
	if ( (nrow == -1) || (nrow > (int)dat.size() ) ) nrow = dat.size();
	Map<VectorXd> vec(&dat[0], nrow);
	return vec;
}

/*
vector<double> dat{ 1, 2, 3, 4, 5, 6 };
ncol = 3;
output matrix:
1 2 3
4 5 6
*/
MatrixXd matGen(vector<double> dat, int nrow) {
	int ncol = (int)(dat.size()) / nrow;
	Map<MatrixXd> mat(&dat[0], ncol, nrow);
	return mat.transpose();
}

/*
dat:
1 2 3 4
5 6 7 8
9 10 11 12
output matrix:
1 2 3 4
5 6 7 8
9 10 11 12
*/
MatrixXd matGen(vector<vector<double>> dat) {
	int nrow = dat.size(), ncol = dat[0].size();
	vector<double> data = flatten_vec2d(dat);
	Map<MatrixXd> mat(&data[0], ncol, nrow);
	return mat.transpose();
}

/*
vec: 1 2 3 4 5 6 7 8 9 10 11 12
output vec1d: 1 2 3 4 5 6 7 8 9 10 11 12
*/
vector<double> vec2vec1d(const VectorXd& vec) {
	vector<double> res(vec.size());
	Map<VectorXd>(res.data(), vec.size()) = vec;
	//Map<MatrixXd>(vec.data(), ncol, nrow) = mat.transpose();
	return res;
}

/*
matrix:
1 2 3
4 5 6
7 8 9
output vec1d:
1 2 3 4 5 6 7 8 9
*/
vector<double> mat2vec1d(const MatrixXd& mat) {
	/* // alternative
	MatrixXd m = mat.transpose();
	vector<double> vec(m.data(), m.data() + m.rows() * m.cols());
	*/
	int ncol = mat.cols(), nrow = mat.rows();
	vector<double> vec(ncol * nrow);
	Map<MatrixXd>(vec.data(), ncol, nrow) = mat.transpose();	
	return vec;
}

vector<vector<double> > mat2vec2d(const MatrixXd& mat) {
	int nrow = mat.rows();
	vector<double> tmp = mat2vec1d(mat);
	return reshape_vec1d(tmp, nrow);
}

void lu_fp_decomp(const MatrixXd& m, MatrixXd& P, MatrixXd& Q, MatrixXd& L, MatrixXd& U) {
	Eigen::FullPivLU<MatrixXd> lu(m);
	L = MatrixXd::Identity(m.rows(), m.rows());
	L.block(0, 0, m.rows(), m.cols()).triangularView<StrictlyLower>() = lu.matrixLU();
	U = lu.matrixLU().triangularView<Upper>();
	P = lu.permutationP();
	Q = lu.permutationQ();
}

void lu_pp_decomp(const MatrixXd& m, MatrixXd& P, MatrixXd& L, MatrixXd& U) {
	Eigen::PartialPivLU<MatrixXd> lu(m);
	L = MatrixXd::Identity(m.rows(), m.rows());
	L.block(0, 0, m.rows(), m.cols()).triangularView<StrictlyLower>() = lu.matrixLU();
	U = lu.matrixLU().triangularView<Upper>();
	P = lu.permutationP();
}


void ldlt_decomp(const MatrixXd& mat, MatrixXd& L, MatrixXd& D) {
	LLT<MatrixXd> llt(mat);
	L = llt.matrixL();
	VectorXd Dv = L.diagonal();
	D = Dv.asDiagonal();
	MatrixXd LT = (L * D.inverse()).transpose();
	L = LT.transpose();
	D *= D;
}

void ldlt_decomp(const MatrixXd& mat) {
	MatrixXd L, D;
	ldlt_decomp(mat, L, D);
	cout << "ldlt_decomp\n"<< "L:\n" << L << endl;
	cout << "D:\n" << D << endl;
}

void ldlt_decomp(vector<vector<double> > dat, vector<vector<double> >& L1, vector<vector<double> >& D1) {
	MatrixXd mat = matGen(dat);
	MatrixXd L, D;
	ldlt_decomp(mat, L, D);
	L1 = mat2vec2d(L);
	D1 = mat2vec2d(D);
}

void ldlt_decomp(vector<vector<double> > dat) {
	MatrixXd mat = matGen(dat);
	ldlt_decomp(mat);	
}