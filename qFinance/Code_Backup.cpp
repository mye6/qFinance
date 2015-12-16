	// 11/3/2015 
	
	//? min-heap solution for 
	// find out whether there exist M days within the last N (N >= M) trading days that 
	// the average closing price of these M days is at most P
	int N = 100, M = 5; float P = 1500.;
	ofstream out("out.txt"); vector<float> prices = read_rev_csv("CSV/GSPC.csv");
	print_vec(prices);
	for (size_t i = 0; i <= prices.size() - N; ++i) {
		vector<float> tmp_vec = sub_vec(prices, i, N);
		out << i << ": " << tmp_vec.back() << ", ";
		if (avg_lower(tmp_vec, M, P)) out << "yes" << endl;
		else out << "no" << endl;
	}
	// note: use vec.back() to find the last element of a vector
	
	//? implement a string indexOf method that returns index of matching string
	
	
	// usage of floor, ceil, log2, pow, log10, exp
	cout << floor(2.5) << endl; // 2
	cout << floor(-3.2) << endl; // -4
	cout << ceil(2.5) << endl; // 3
	cout << ceil(-3.2) << endl; // -3
	cout << log2(4) << endl; // 2
	cout << pow(2, 3) << endl; // 8
	cout << log10(100) << endl; // 2
	cout << exp(1) << endl; // 2.71828
	
	//? write a function to calculate power(x, n)
	cout << pow(2., 0) << ", " << power_rec(2., 0) << endl;
	cout << pow(2., 3) << ", " << power_rec(2., 3) << endl;
	cout << pow(-2., 3) << ", " << power_rec(-2., 3) << endl;
	cout << pow(2.2, 3) << ", " << power_rec(2.2, 3) << endl;
	cout << pow(-2.2, 4) << ", " << power_rec(-2.2, 4) << endl;
	cout << pow(-2.2, 1) << ", " << power_rec(-2.2, 1) << endl;
	
	// note: n++ where n is a double, still works
	double n = 1.0;
	cout << n << endl; // 1
	n++;
	cout << n << endl; // 2
	
	//? write a function to calculate exp(x)
	cout << numeric_limits<double>::min() << endl; // 2.22507e-308
	cout << exp(0.) << ", " << exp_taylor(0.) << endl; // 1, 1
	cout << exp(1.) << ", " << exp_taylor(1.) << endl; // 2.71828, 2.71828
	cout << exp(2.3) << ", " << exp_taylor(2.3) << endl; // 9.97418, 9.97418
	cout << exp(10.2) << ", " << exp_taylor(10.2) << endl; // 26903.2, 26903.2
	cout << exp(-10.2) << ", " << exp_taylor(-10.2) << endl; // 3.71703e-005, 3.71703e-005
	
	//? calculate the first 100 prime numbers
	vector<int> primes = prime_vec(100);
	print_vec<int>(primes,primes.size());
	
	//? unif and normal random generator
	ofstream out1("output/normal.dat");
	ofstream out2("output/unif.dat");
	for (int i = 0; i < 10000; ++i) {
		out1 << normal() << endl;
		out2 << unif() << endl;
	}
	
	file = "../output/normal.dat"
	y=scan(file)
	qqnorm(y)
	qqline(y)

	file = "../output/unif.dat"
	y=scan(file)
	hist(y)
	hist(y,breaks=100)
	
	//? online computation of median 
	
	
	// note: initialize a vector
	vector<double> vec(4);
	vec = {3., 2.2, 5., 3.5};
	sort(vec.begin(), vec.end());
	print_vec<double>(vec);
	
	// note: find the value in the vector ranges
	// std::upper_bound
	// Returns an iterator pointing to the first element in the range [first,last) which compares greater than val.
	vector<double> vec{ 1, 2, 3, 4, 5 };
	print_vec<double>(vec);
	int B = vec.size();
	int lower, upper;
	double val;
	for (int i = 0; i < 12; ++i) {
		val = (double)i / 2.;
		lower = (lower_bound(vec.begin(), vec.end(), val) - vec.begin());
		upper = (upper_bound(vec.begin(), vec.end(), val) - vec.begin());
		cout << val << " -> lower, " << lower << "; upper " << upper << endl;
	}
	/*
	Output:
	0 -> lower, 0; upper 0
	0.5 -> lower, 0; upper 0
	1 -> lower, 0; upper 1
	1.5 -> lower, 1; upper 1
	2 -> lower, 1; upper 2
	2.5 -> lower, 2; upper 2
	3 -> lower, 2; upper 3
	3.5 -> lower, 3; upper 3
	4 -> lower, 3; upper 4
	4.5 -> lower, 4; upper 4
	5 -> lower, 4; upper 5
	5.5 -> lower, 5; upper 5	
	*/
	
	// online median processor
	vector<double> vec = read_rev_csv("CSV/BBY.csv");
	int N = 100;
	ofstream out("output/med.dat");
	OnlineQuantileB oq(sub_vec(vec, 0, N));
	for (size_t i = N; i < vec.size(); ++i){
		oq.addNumber(vec[i]);
		out << oq.getValue() << endl;
	}
	
	// note: difference const_iterator, iterator
	// The const_iterator returns a reference to a constant value (const T&) and prevents modification of the referenced value: it enforces const-correctness. When you have a const reference to the container, you can only get a const_iterator
	
	// test dataframe
	cout << "main print_vec" << endl;
	print_vec<double>(df.getCol("X"));
	cout << "dim()" << endl;
	print_vec<int>(df.dim());
	print_vec<string>(df.keys());
	string ofp = "CSV/dd_out.csv";
	df.to_csv(ofp);
	
	string ofp = "CSV/dd_out2.csv";
	DataFrame df2(df.keys(), df.data());
	df2.to_csv(ofp);
	
	string fp = "CSV/dd.csv";
	DataFrame df(fp);
	
	string fp2 = "CSV/dd_out2.csv";
	DataFrame df2(fp2);

	string ofp1 = "CSV/dd_out3.csv";
	df2.to_csv(ofp1);

	string ofp2 = "CSV/dd_out4.csv";
	df2 = df;
	df2.to_csv(ofp2);

	// usage of DataFrame
	vector<string> keys{"x", "y"};
	vector<double> X{ 1., 2., 3., 4., 5. };
	vector<double> Y{ 2., 3., 0.5, 2., -1 };
	vector<vector<double> > dat;
	dat.push_back(X); dat.push_back(Y);
	DataFrame df(keys, dat);
	string out_path = "output/xy.csv";
	df.to_csv(out_path);
	
	// Eigen library, solve linear system equation
	Matrix3f A;
	Vector3f b;
	A << 1, 2, 3, 4, 5, 6, 7, 8, 10;
	b << 3, 3, 4;
	cout << "Here is the matrix A:\n" << A << endl;
	cout << "Here is the vector b:\n" << b << endl;
	Vector3f x = A.colPivHouseholderQr().solve(b);
	cout << "The solution is:\n" << x << endl;
	
	Matrix3f A;
	Vector3f b;
	A << 15, 11, 19, 11, 36, 3, 19, 3, 31;
	b << -3, 6, -5;
	cout << "Here is the matrix A:\n" << A << endl;
	cout << "Here is the vector b:\n" << b << endl;
	Vector3f x = A.colPivHouseholderQr().solve(b);
	cout << "The solution is:\n" << x << endl;
	/*
	output:
	Here is the matrix A:
	15 11 19
	11 36  3
	19  3 31
	Here is the vector b:
	-3
	 6
	-5
	The solution is:
	-1.59173
	0.589928
	0.757194	
	*/
	
	
	
	// Solve linear system equations
	/*
	1. Gauss
	
	
	*/
	
	Matrix3d A;
	Vector3d b;
	//A << 1, -1, 1, -1, 3, -2, 1, -2, 4.5;
	A << 4, 12, -16, 12, 37, -43, -16, -43, 98;
	//b << -3, 6, -5;	

	cout << "Here is the matrix A:\n" << A << endl;

	Matrix3d Ld(A.ldlt().matrixL());
	cout << "Directly, Here is the matrix Ld:\n" << Ld << endl;

	LDLT<MatrixXd> ldlt(A);
	MatrixXd L0 = ldlt.matrixL();
	cout << "Previously, Here is the matrix L:\n" << L0 << endl;

	LLT<MatrixXd> llt(A);
	MatrixXd L = llt.matrixL();
	cout << "Previously, Here is the matrix L:\n" << L << endl;

	VectorXd Dv = L.diagonal();
	//cout << "Here is the vector D:\n" << Dv << endl;

	MatrixXd D = Dv.asDiagonal();
	// cout << "Here is the vector D as matrix: \n" << D << endl;

	MatrixXd LT = (L * D.inverse()).transpose(); 	

	L = LT.transpose();
	cout << "Here is the matrix L:\n" << L << endl;

	D *= D;
	cout << "Here is the matrix D: \n" << D << endl;

	double det = A.determinant();
	cout << "Determinant of matrix A: \n" << det << endl;
	
	// convert vector
	double data[] = { 1, 2, 3, 4 ,5, 6};
	Map<MatrixXd> m2(data, 2, 3);
	vector<double> data2{ 1, 2, 3, 4, 5, 6 };
	cout << "Here is the matrix data2:\n" << matGen(data2, 3) << endl;
	
	vector<double> data2{ 1, 2, 3};
	cout << "Here is the v1:\n" << vecGen(data2) << endl;
	
	vector<double> data2{ 1, 2, 3, 4, 5, 6 };
	MatrixXd AA = matGen(data2, 3);
	vector<double> v2(2 * 3);
	Map<MatrixXd>(v2.data(), 3, 2) = AA.transpose();
	cout << "v2:" << endl;
	print_vec<double>(v2);
	
	
	/*
	vector<double> vec0{ 4., 12., -16 };
	vector<double> vec1{ 12., 37., -43.};
	vector<double> vec2{ -16., -43., -98. };
	*/
	vector<double> vec0{ 1., 2., 3. };
	vector<double> vec1{ 4., 5., 6. };
	vector<double> vec2{ 7., 8., 9. };
	vector<vector<double> > dat{vec0, vec1, vec2};

	print_vec2d<double>(dat);
	
	
	int m = dat.size(), n = dat[0].size();
	Matrix3d A(m, n);
	for (int i = 0; i < m; ++i)
		for (int j = 0; j < n; ++j)
			A(i, j) = dat[i][j];
	cout << "Here is the matrix A:\n" << A << endl;

	print_vec<double>(flatten_vec2d(dat));

	//double data[] = { 1, 2, 3, 4};
	// vector<double> data2{ 1, 2, 3};	
	
	//Map<VectorXd> v1(&data2[0], 4);       // uses v1 as a Vector3f object
	// Map<ArrayXf>  v2(data, 3);     // uses v2 as a ArrayXf object
	// Map<Array22f> m1(data);       // uses m1 as a Array22f object
	// cout << "Here is the v1:\n" << vecGen(data2) << endl;
	
	vector<double> data2{ 1, 2, 3, 4, 5, 6 };
	MatrixXd AA = matGen(data2, 3);
	cout << "Here is the matrix AA:\n" << AA << endl;
	cout << "Here is the vector from AA:\n";
	print_vec<double>(mat2vec1d(AA));

	vector<vector<double> > res;
	res = reshape_vec1d(data2, 2);
	/*	
	for (int i = 0; i < 2; ++i) {
		res.push_back(sub_vec(data2, i*3, 3));
	}
	*/
	//res.resize(3, data2);
	cout << "res:2d" << endl;
	print_vec2d<double>(res);


	//cout << "ldlt decomp" << endl; 
	//ldlt_decomp(dat);
	/*
	for (size_t i = 0; i < dat.size(); ++i) {
		for (size_t j = 0; j < dat[i].size() - 1; ++j) {
			cout << dat[i][j] << " ";
		}
		cout << dat[i][dat[i].size()-1] << endl;
	}
	*/
	
	// function pointer
	ofstream out0("output/f.txt");
	ofstream out("output/fprime.txt");
	out0 << "x y" << endl;
	out << "x yp" << endl;
	for (int i = -50; i <= 50; ++i) {
		double x = 0.1*i;
		out0 << x << " " << sin(x) << endl;
		out << x << " " << fprime(msine, x) << endl;
	}
	
	double fprime(const std::function<double(double)> &f, double x0, double eps) {
		// calculate the derivative via f'(x) = (f(x + h/2) - f(x - h/2))/h
		double eps2 = eps / 2, lo = x0 - eps2, hi = x0 + eps2;
		return (f(hi) - f(lo)) / eps;
	}
	
	
	// ostream
	vector<double> tmp{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }; 
	vector<vector<double> > dat = reshape_vec1d(vector<double>{1,2,3,4,5,6,7,8,9,10,11,12}, 3);
	cout << "tmp: " << endl;
	cout << tmp << endl;
	cout << "dat: " << endl;
	cout << dat << endl;
	
	/*
	tmp:
	1 2 3 4 5 6 7 8 9 10 11 12
	dat:
	1 2 3 4
	5 6 7 8
	9 10 11 12
	*/
	
	vector<double> vec{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	vector<vector<double> > dat = reshape_vec1d(vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 3);
	cout << "dat:\n" << dat << endl;
	VectorXd vecx = vecGen(vec);	
	cout << "vecx:\n" << vecx << endl;
	MatrixXd matx1 = matGen(vec, 2);
	cout << "matx1:\n" << matx1 << endl;
	MatrixXd matx2 = matGen(dat);
	cout << "matx2:\n" << matx2 << endl;
	vector<double> v0 = vec2vec1d(vecx);
	cout << "v0:\n" << v0 << endl;
	vector<double> v1 = mat2vec1d(matx1);
	cout << "v1:\n" << v1 << endl;
	vector<vector<double> > v2 = mat2vec2d(matx1);
	cout << "v2:\n" << v2 << endl;
	vector<vector<double> > v3 = mat2vec2d(matx2);
	cout << "v3:\n" << v3 << endl;
	/*
	output:
	 1  2  3  4
	 5  6  7  8
	 9 10 11 12
	v0:
	1 2 3 4 5 6 7 8 9 10 11 12
	v1:
	1 2 3 4 5 6 7 8 9 10 11 12
	v2:
	1 2 3 4 5 6
	7 8 9 10 11 12
	v3:
	1 2 3 4
	5 6 7 8
	9 10 11 12
	*/
	
	cout << "mat:\n" << mat << endl;
	MatrixXd L, D;	
	ldlt_decomp(mat, L, D);
	cout << "L:\n" << L << endl;
	cout << "D:\n" << D << endl;
	ldlt_decomp(mat);
	cout << "dat---" << endl;
	ldlt_decomp(dat);
	
	
	typedef Matrix<double, 5, 3> Matrix5x3;
	typedef Matrix<double, 5, 5> Matrix5x5;
	Matrix5x3 m = Matrix5x3::Random();
	cout << "Here is the matrix m:" << endl << m << endl;
	Eigen::FullPivLU<Matrix5x3> lu(m);
	//cout << "Here is, up to permutations, its LU decomposition matrix:"
		//<< endl << lu.matrixLU() << endl;
	//cout << "Here is the L part:" << endl;
	Matrix5x5 l = Matrix5x5::Identity();
	l.block<5, 3>(0, 0).triangularView<StrictlyLower>() = lu.matrixLU();
	// l = lu.matrixLU();
	//cout << l << endl;
	//cout << "Here is the U part:" << endl;
	Matrix5x3 u = lu.matrixLU().triangularView<Upper>();
	//cout << u << endl;
	cout << "Let us now reconstruct the original matrix m:" << endl;
	cout << lu.permutationP().inverse() * l * u * lu.permutationQ().inverse() << endl;
	
	cout << "Here is the L part:\n" << l << endl;
	cout << "Here is the U part:\n" << u << endl;
	MatrixXd P = lu.permutationP();
	cout << "Here is the P part:\n" << P << endl;
	MatrixXd Pinv = lu.permutationP().inverse();
	cout << "Here is the Pinv part:\n" << Pinv << endl;
	MatrixXd Q = lu.permutationQ();
	cout << "Here is the Q part:\n" << Q << endl;
	MatrixXd Qinv = lu.permutationQ().inverse();
	cout << "Here is the Qinv part:\n" << Qinv << endl;
	
	VectorXd X(3);
	X << -2, 3, -1;
	//double nn = X.lpNorm<1>();
	cout << X.lpNorm<1>() << endl;
	
	MatrixXd m(3, 3);
	// m << 1, 2, 3, 3, 5, 7, 2, 9, 5;
	m << 4, 12, -16, 12, 37, -43, -16, -43, 89;
	int nrow = 3, ncol = 3;

	//int nrow = 6, ncol = 4;
	//MatrixXd m = MatrixXd::Random(nrow, ncol);

	cout << "Here is the matrix m:" << endl << m << endl;
	Eigen::FullPivLU<MatrixXd> lu(m);
	MatrixXd l = MatrixXd::Identity(nrow, nrow);
	l.block(0, 0, nrow, ncol).triangularView<StrictlyLower>() = lu.matrixLU();
	cout << "l\n" << l << endl;
	MatrixXd u = lu.matrixLU().triangularView<Upper>();
	cout << "u\n" << u << endl;
	MatrixXd P = lu.permutationP();
	MatrixXd Q = lu.permutationQ();
	MatrixXd mr = P.inverse()*l*u*Q.inverse();
	cout << "P\n" << P << endl;
	cout << "Q\n" << Q << endl;
	cout << "mr\n" << mr << endl;
	//double norm1 = mr.norm();
	//cout << "norm1: " << norm1 << endl;
	cout << "relative error:\n" << (m - mr).norm() / m.norm() << endl;

	cout << "Here is, up to permutations, its LU decomposition matrix:" << endl << lu.matrixLU() << endl;
	cout << "P*M*Q:" << endl << P*m*Q << endl;

	{
		MatrixXd m(3, 3);
		m << 1, 2, 3, 3, 5, 7, 2, 9, 5;
		int nrow = 3, ncol = 3;

		//int nrow = 6, ncol = 4;
		//MatrixXd m = MatrixXd::Random(nrow, ncol);

		cout << "Here is the matrix m:" << endl << m << endl;
		Eigen::PartialPivLU<MatrixXd> lu(m);
		MatrixXd l = MatrixXd::Identity(nrow, nrow);
		l.block(0, 0, nrow, ncol).triangularView<StrictlyLower>() = lu.matrixLU();
		cout << "l\n" << l << endl;
		MatrixXd u = lu.matrixLU().triangularView<Upper>();
		cout << "u\n" << u << endl;
		MatrixXd P = lu.permutationP();		
		MatrixXd mr = P.inverse()*l*u;
		cout << "P\n" << P << endl;		
		cout << "mr\n" << mr << endl;
		//double norm1 = mr.norm();
		//cout << "norm1: " << norm1 << endl;
		cout << "relative error:\n" << (m - mr).norm() / m.norm() << endl;

		cout << "Here is, up to permutations, its LU decomposition matrix:" << endl << lu.matrixLU() << endl;
		cout << "l*u:" << endl << l*u << endl;
	}

	{
		//example 1
		typedef Eigen::Matrix<double, 4, 4> M4x4;
		M4x4 p;
		p << 7, 3, -1, 2, 3, 8, 1, -1, -1, 1, 4, -1, 2, -4, -1, 6;
		cout << p << endl << endl;
		// Create LU Decomposition template object for p
		Eigen::PartialPivLU<M4x4> LU(p);
		cout << "LU MATRIX:\n" << LU.matrixLU() << endl << endl;
		// Output L, the lower triangular matrix
		M4x4 l = MatrixXd::Identity(4, 4);//默认 单位对角矩阵
		//开始填充
		l.block<4, 4>(0, 0).triangularView<Eigen::StrictlyLower>() = LU.matrixLU();
		cout << "L MATRIX:\n" << l << endl << endl;
		M4x4 u = LU.matrixLU().triangularView<Eigen::Upper>();
		cout << "R MATRIX:\n" << u << endl << endl;
		MatrixXd m0(4, 4);
		m0 = l*u;
		cout << "calculate the original matrix:\n" << m0 << endl << endl;
		//证明完毕
		//example 2
		typedef Eigen::Matrix<double, 2, 2> M2X2;
		M2X2 p0;
		p0 << 0, 2, 1, 3;
		cout << p0 << endl << endl;
		Eigen::PartialPivLU<M2X2> LU0(p0);
		cout << "LU MATRIX:\n" << LU0.matrixLU() << endl << endl;//原来是在做PA的过程
		//一切结果从PA开始
		M2X2 l0 = MatrixXd::Identity(2, 2);
		l0.block<2, 2>(0, 0).triangularView<Eigen::StrictlyLower>() = LU0.matrixLU();
		cout << "L MATRIX:\n" << l0 << endl << endl;
		//以下省略N行
		//example 3
		typedef Eigen::Matrix<double, 3, 3> M3X3;
		M3X3 p1;
		p1 << 3, -1, 2, 6, -1, 5, -9, 7, 3;
		cout << p1 << endl << endl;
		Eigen::PartialPivLU<M3X3> LU1(p1);
		cout << "LU MATRIX:\n" << LU1.matrixLU() << endl << endl;//暂时没明白这步做的啥
		M3X3 l1 = MatrixXd::Identity(3, 3);
		l1.block<3, 3>(0, 0).triangularView<Eigen::StrictlyLower>() = LU1.matrixLU();
		cout << "L MATRIX:\n" << l1 << endl << endl;
		//直接up
		M3X3 u1 = LU1.matrixLU().triangularView<Eigen::Upper>();
		cout << "R MATRIX:\n" << u1 << endl << endl;
		cout << l1*u1 << endl;
		cin.get();
	}
	
	
	MatrixXd m(3, 3);	
	m << 4, 12, -16, 12, 37, -43, -16, -43, 89;
	MatrixXd P, Q, L, U;
	lu_fp_decomp(m, P, Q, L, U);
	cout << "L:\n" << L << endl;
	cout << "U:\n" << U << endl;
	cout << "P:\n" << P << endl;
	cout << "Q:\n" << Q << endl;
	cout << "M\n" << m << endl;
	cout << "Pi*L*U*Qi\n" << P.inverse()*L*U*Q.inverse() << endl;
	cout << "P*M*Q\n" << P*m*Q << endl;
	cout << "L*U\n" << L*U << endl;
	
	MatrixXd P, L, U;
	lu_pp_decomp(m, P, L, U);
	cout << "L:\n" << L << endl;
	cout << "U:\n" << U << endl;
	cout << "P:\n" << P << endl;
	cout << "M\n" << m << endl;
	cout << "P*M\n" << P*m << endl;
	cout << "Pi*L*U\n" << P.inverse()*L*U << endl;
	cout << "L*U\n" << L*U << endl;
	
	int m = 4;
	int n = 3;

	MatrixXcd A (m,n), Q, R;
	A.setRandom();
	
	
	
	
	MatrixXd A(4, 3);	
	A << 4, 12, -16, 12, 37, -43, -16, -45, 78, 4, 5, 6;
	Eigen::HouseholderQR<MatrixXd> qr(A);
	MatrixXd Q, R;
	
	cout << "A:\n" << A << endl;
	
	
	Q = qr.householderQ();
	R = qr.matrixQR().triangularView<Upper>();

	
	cout << "Q:\n" << Q << endl;
	cout << "R:\n" << R << endl;
	
	cout << "Q*R\n" << Q*R << endl;
	cout << "Q*QT\n" << Q*Q.transpose() << endl;
	
	
	
	// integration
	PRINT(Integral::leftRect(mymath::f1, 1., 3.));
	PRINT(Integral::rightRect(mymath::f1, 1., 3.));
	PRINT(Integral::midRect(mymath::f1, 1., 3.));
	PRINT(Integral::trapezium(mymath::f1, 1., 3.));
	PRINT(Integral::simpson(mymath::f1, 1., 3.));
	
	// differentiation
	PRINT(Differentiation::leftDiff(mymath::f1, .5));
	PRINT(Differentiation::rightDiff(mymath::f1, .5));
	PRINT(Differentiation::centDiff(mymath::f1, .5));
	PRINT(Differentiation::centDiff2(mymath::f1, .5));
	
	// pointer
	int a = 23;
	int* ipa = &a;
	PRINT(a);
	PRINT(&a);
	PRINT(ipa);
	PRINT(*ipa);
	/*
	a: 23
	&a: 0018FBB0
	ipa: 0018FBB0
	*ipa: 23
	*/
	
	DataFrame df("output/f.txt", ' ');
	PRINT(df.keys());
	SEP;
	PRINT(df.data());
	PRINT(df.getCol("x"));
	PRINT(df.getCol("y"));
	PRINT(df.dim());
	
	cout << sizeof(unsigned char) << endl;
	for (int v = 0; v <= 255; ++v) {
		cout << "v: " << v << ", ";
		PRINT(toBinary((unsigned char)v)); // auto
	}
	
	// function pointer
	#include "Solver.h"
	void hello() { cout << "hello" << endl; }
	void goodbye() { cout << "goodbye" << endl; }
	typedef void(*FunctionPointerVoid)();

	double add(double a, double b) { return a + b; }
	double multiply(double a, double b) { return a * b; }
	double substract(double a, double b) { return a - b; }
	double dividedby(double a, double b) { return a / b; }
	typedef double (*FunctionPointer2d)(double, double);

	double testFunctionPointer(FunctionPointerVoid& fpv, FunctionPointer2d& fp2, double a = 10., double b = 5.) {
		(*fpv)();
		return (*fp2)(a, b);
	}

	int main() {	
		double a = 50., b = 20.;
		cout << "testFunctionPointer:" << endl;	
		FunctionPointerVoid fpv = hello;
		FunctionPointer2d fp2 = multiply;
		cout << testFunctionPointer(fpv, fp2, a, b) << endl;	
		system("pause");
		return 0;
	}
	
	// dsaa::IntCell class
	dsaa::IntCell obj(37);
	dsaa::IntCell o2(obj);
	PRINT(obj.read());
	PRINT(o2.read());
	vector<dsaa::IntCell> vec(5);
	for (size_t i = 0; i < vec.size(); ++i)
		PRINT(vec[i].read());

	dsaa::IntCell* m = new dsaa::IntCell(5);
	PRINT(m->read());
	m->write(6);
	PRINT(m->read());
	delete m;
	
	SEP;
	dsaa::IntPtCell pobj(37);
	dsaa::IntPtCell po2(pobj);
	PRINT(pobj.read());
	PRINT(po2.read());
	vector<dsaa::IntPtCell> pvec(5);
	for (size_t i = 0; i < pvec.size(); ++i)
		PRINT(pvec[i].read());

	dsaa::IntPtCell* pm = new dsaa::IntPtCell(5);
	PRINT(pm->read());
	pm->write(6);
	PRINT(pm->read());
	delete pm;
	
	// findMax
	vector<int> v{ 1, 3, 2, 6, 3 };
	PRINT(dsaa::findMax(v));

	dsaa::findMax<int>(v) = 10;
	PRINT(v);
	
	// Namespaces and Operator Overloading in C++
	/*
	When authoring a library in a particular namespace, it's often convenient to provide overloaded operators for the classes in that namespace. It seems (at least with g++) that the overloaded operators can be implemented either in the library's namespace:
	A: You should define them in the library namespace. The compiler will find them anyway through argument dependant lookup.
	URL: http://stackoverflow.com/questions/171862/namespaces-and-operator-overloading-in-c
	See: DSAA, Employee Class *******************************IMPORTANT!
	*/
	vector<dsaa::Employee> v(3);
	v[0].setValue("GS", 400000.0);
	v[1].setValue("Bill Gates", 2000000.0);
	v[2].setValue("Dr. Phil", 1200000000.0);
	cout << dsaa::findMax(v) << endl;
	
	// friend ostream& operator<<(ostream& os, const Vector& vec);
	// dsaa::operator<< unresolve
	dsaa::Vector<double> vec(10, 1.);
	dsaa::Vector<double> vec2(vec);
	vec = vec2;
	vec.reserve(10);	
	vec[1] = 2.;
	vec[2] = 3.;
	cout << vec[0] << ", " << vec[1] << ", " << vec[2] << endl;
	cout << vec.size() << ", " << vec.capacity() << ", " << vec[2] << endl;
	vec.push_back(9);
	cout << vec.size() << ", " << vec.capacity() << ", " << vec[2] << endl;
	cout << vec.empty() << endl;
	cout << vec.back() << endl;
	dsaa::Vector<double>::iterator b0 = vec.begin();
	dsaa::Vector<double>::const_iterator b1 = vec.begin();
	PRINT(*b0);
	PRINT(*b1);
	*b0 = 2.0;
	PRINT(*b0);
	PRINT(*b1);
	vec.print();
	
	// template<T> ostream& operator<<(ostream& os, const Vector& vec)
	just define the friend function inside the class, no need to move out as the compiler will complain
	see: dsaa::Employee operator << 
	
	// Financial Recipes
	{
		dsaa::List<double> ls;
		ls.push_back(3.5);
		ls.push_back(2.5);
		ls.push_back(1.5);
		ls.push_front(0.5);
		cout << ls << endl;
		dsaa::List<double> ls1(ls);
		cout << ls.front() << endl;
		cout << ls.back() << endl;
		cout << ls.size() << ls.empty() << endl;		
		ls.clear();
	}
	
	cout << pow(2., 3.) << endl;
	
	
	
	
	//? bond pricing and IRR
	vector<double> times{ 0., 1., 2. };
	vector<double> cflows{ -100., 10, 110 };
	double r = 0.05;
	PRINT(cash_flow_pv_discrete(times, cflows, r));
	PRINT(cash_flow_pv(times, cflows, r));
	PRINT(cash_flow_unique_irr(times, cflows));
	PRINT(cash_flow_irr_discrete(times, cflows));
	
	// Futures
	vector<double> times{ 1., 2., 3. };
	vector<double> cflows{ 10., 10., 110 };
	double r = 0.09;
	PRINT(bonds_price_discrete(times, cflows, r));
	double bondprice = bonds_price_discrete(times, cflows, r);
	PRINT(bonds_yield_to_maturity_discrete(times, cflows, bondprice));
	PRINT(bonds_duration_discrete(times, cflows, r));
	PRINT(bonds_duration_modified_discrete(times, cflows, bondprice));
	PRINT(bonds_convexity_discrete(times, cflows, r));
	SEP;
	PRINT(bonds_price(times, cflows, r));	
	PRINT(bonds_yield_to_maturity(times, cflows, bondprice));
	PRINT(bonds_duration(times, cflows, r));
	PRINT(bonds_duration_macaulay(times, cflows, bondprice));
	PRINT(bonds_convexity(times, cflows, r));
	
	// functor
	A functor is pretty much just a class which defines the operator(). That lets you create objects which "look like" a function:
	// this is a functor
	struct add_x {
	  add_x(int x) : x(x) {}
	  int operator()(int y) const { return x + y; }
	private:
	  int x;
	};
	// Now you can use it like this:
	add_x add42(42); // create an instance of the functor class
	int i = add42(8); // and "call" it
	assert(i == 50); // and it added 42 to its argument
	
	// futures
	double S = 100, r = 0.1, time = 0.5;
	PRINT(futures_price(S, r, time));
	
	// Black Scholes, calculation
	// check: url: http://www.soarcorp.com/black_scholes_calculator.jsp
	double S = 50., K = 50., r = 0.1, sigma = 0.3, time = 0.5;
	PRINT(OptionPricer_BS::call(S, K, r, sigma, time));
	PRINT(OptionPricer_BS::put(S, K, r, sigma, time));

	double Delta, Gamma, Theta, Vega, Rho;
	OptionPricer_BS::call_partials(
		S, K, r, sigma, time, Delta, Gamma, Theta, Vega, Rho);
	SEP;
	PRINT(Delta); PRINT(Gamma); PRINT(Theta); PRINT(Vega); PRINT(Rho);

	OptionPricer_BS::put_partials(
		S, K, r, sigma, time, Delta, Gamma, Theta, Vega, Rho);
	SEP;
	PRINT(Delta); PRINT(Gamma); PRINT(Theta); PRINT(Vega); PRINT(Rho);
	
	// output	
	OptionPricer_BS::call(S, K, r, sigma, time): 5.45325
	OptionPricer_BS::put(S, K, r, sigma, time): 3.01472
	------------------------------------------------------------
	Delta: 0.633737
	Gamma: 0.0354789
	Theta: -6.61473
	Vega: 13.3046
	Rho: 13.1168
	------------------------------------------------------------
	Delta: -0.366263
	Gamma: 0.0354789
	Theta: -1.85859
	Vega: 13.3046
	Rho: -10.6639
	
	
	// Options pricing
	double S = 50., K = 50., r = 0.1, time = 0.5;
	double C = 2.5;	
	PRINT(ImpliedVol_BS::call_newton(S, K, r, time, C));
	PRINT(ImpliedVol_BS::call_bisections(S, K, r, time, C));

	double P = 0.061;
	PRINT(ImpliedVol_BS::put_newton(S, K, r, time, P));
	PRINT(ImpliedVol_BS::put_bisections(S, K, r, time, P));
	
	
	double S = 100., K = 100., r = 0.1, time = 1., q = 0.05, sigma = 0.25;	
	PRINT(option_price_european_call_payout(S, K, r, q, sigma, time));
	vector<double> dividend_times{0.25, 0.75};
	vector<double> dividend_amounts{ 2.5, 2.5 };
	PRINT(option_price_european_call_dividends(S, K, r, sigma, time, dividend_times, dividend_amounts));
	
	// American call options
	double S = 100., K = 100., r = 0.1, sigma = 0.25;
	double tau = 1.0, tau1 = 0.5, D1 = 10.0;
	PRINT(option_price_american_call_one_dividend(S, K, r, sigma, tau, D1, tau1));


	S = 50.0, K = 40., r = 0.05, sigma = 0.05;
	double q = 0.02;
	PRINT(option_price_american_perpetual_call(S, K, r, q, sigma));
	
	double F = 50., K = 45., r = 0.08, sigma = 0.2, time = 0.5;
	
	PRINT(futures_option_price_call_european_black(F, K, r, sigma, time));
	SEP;


	PRINT(futures_option_price_put_european_black(F, K, r, sigma, time));
	cout << "here" << endl;
	
	double S = 50., K = 52., r = 0.08, rf = 0.05, sigma = 0.2, time = 0.5;
	PRINT(currency_option_price_call_european_black(S, K, r, rf, sigma, time));
	
	double S = 50., K = 50., r = 0.1, sigma = 0.3, time = 0.5;
	double Delta, Gamma, Theta, Vega, Rho;
	option_price_partials_call_black_scholes(
		S, K, r, sigma, time, Delta, Gamma, Theta, Vega, Rho);
	SEP;
	PRINT(Delta); PRINT(Gamma); PRINT(Theta); PRINT(Vega); PRINT(Rho);

	option_price_partials_put_black_scholes(
		S, K, r, sigma, time, Delta, Gamma, Theta, Vega, Rho);
	SEP;
	PRINT(Delta); PRINT(Gamma); PRINT(Theta); PRINT(Vega); PRINT(Rho);
	
	
	double S = 50., K = 50., r = 0.1, time = 0.5;
	double C = 2.5;
	PRINT(option_price_implied_volatility_call_black_scholes_newton(S, K, r, time, C));
	PRINT(option_price_implied_volatility_call_black_scholes_bisections(S, K, r, time, C));

	double P = 0.061;
	PRINT(option_price_implied_volatility_put_black_scholes_newton(S, K, r, time, P));
	PRINT(option_price_implied_volatility_put_black_scholes_bisections(S, K, r, time, P));
	
	
	
	
	double S = 100., K = 100., r = 0.1, sigma = 0.25, time = 1.0;
	int N = 100;
	PRINT(option_price_call_european_binomial(S, K, r, sigma, time, N));
	PRINT(option_price_put_european_binomial(S, K, r, sigma, time, N));
	PRINT(option_price_call_american_binomial(S, K, r, sigma, time, N));
	PRINT(option_price_put_american_binomial(S, K, r, sigma, time, N));
	
	
	double S = 100., K = 100., r = 0.1, sigma = 0.25, time = 1.0;
	int N = 100;
	double delta, gamma, theta, vega, rho;
	option_price_partials_american_call_binomial(S,K,r,sigma,time,delta,gamma, theta,vega,rho, N);
	PRINT(delta);
	PRINT(gamma);
	PRINT(theta);
	PRINT(vega);
	PRINT(rho);
	SEP;
	option_price_partials_american_put_binomial(S, K, r, sigma, time, delta, gamma, theta, vega, rho, N);
	PRINT(delta);
	PRINT(gamma);
	PRINT(theta);
	PRINT(vega);
	PRINT(rho);
	
	
	
	double Expiry = 2.0, Strike = 20., Spot = 30., Vol = 0.2, r = 0.05;
	unsigned long NumberOfPaths = 1000;
	PayOffCall thePayOff(Strike);
	VanillaOption theOption(thePayOff, Expiry);
	ParametersConstant VolParam(Vol);
	ParametersConstant rParam(r);
	StatisticsMean gatherer;
	ConvergenceTable gathererTwo(gatherer);
	SimpleMonteCarlo(theOption,
		Spot,
		VolParam,
		rParam,
		NumberOfPaths,
		gathererTwo);
	vector<vector<double> > results = gathererTwo.GetResultsSoFar();
	cout << "\nFor the call price the results are \n";
	for (unsigned long i = 0; i < results.size(); i++) {
		for (unsigned long j = 0; j < results[i].size(); j++)
			cout << results[i][j] << " ";
		cout << "\n";
	}
	
	// An example of tail recursive function
void print(int n) {
	if (n < 0)  return;
	cout << " " << n;	
	print(n - 1); // The last executed statement is recursive call, and not used in print(n)
}

// A NON-tail-recursive function.  The function is not tail
// recursive because the value returned by fact(n-1) is used in
// fact(n) and call to fact(n-1) is not the last thing done by fact(n)
unsigned int fact(unsigned int n) {
	if (n == 0) return 1;
	return n*fact(n - 1);
}

Employee empl("John Burke", 25.0);
	Manager mgr("Jan Kovacs", 1200.0, true);
	Supervisor sup("Denise Zephyr", 780.0, "Accounting");

	// Assume all employees worked 40 hours this period.

	cout << "For Employee:" << endl;
	cout << "Name: " << empl.getName() << endl;
	cout << "Pay: " << empl.pay(40.0) << endl;

	cout << "Changing the Employee's name..." << endl;
	empl.setName("Doug Conners");
	cout << "New Name: " << empl.getName() << endl;

	cout << endl;
	cout << "For Manager:" << endl;
	cout << "Name: " << mgr.getName() << endl;
	cout << "Salaried: " << mgr.getSalaried() << endl;
	cout << "Pay: " << mgr.pay(40.0) << endl;

	cout << "Changing the Manager's salaried status..." << endl;
	mgr.setSalaried(false);
	cout << "New Pay: " << mgr.pay(40.0) << endl;

	cout << endl;
	cout << "For Supervisor:" << endl;
	cout << "Name: " << sup.getName() << endl;
	cout << "Pay: " << sup.pay(40.0) << endl;
	cout << "Dept: " << sup.getDept() << endl;

	cout << "Changing the Supervisor's pay rate..." << endl;
	sup.setPayRate(900.0);
	cout << "New Pay: " << sup.pay(40.0) << endl;le
	
	
	// leetcode
	vector<int> nums{2, 1, 3, 5};
	PRINT(Solution::rob(nums));
	
	for (int i = 1; i < 10; ++i) {
		PRINT(i);
		PRINT(Solution::climbStairs(i));
		SEP;
	}
	
	vector<TreeNode *> vec = Solution::generateTrees(3);
	for (size_t i = 0; i < vec.size(); ++i) {
		Solution::printTree(vec[i]);
		SEP;
	}
	
	
	TreeNode* root = new TreeNode(1);
	root->left = new TreeNode(2);
	root->right = new TreeNode(3);
	root->left->left = new TreeNode(4);
	root->right->left = new TreeNode(5);
	root->right->right = new TreeNode(6);
	root->right->left->right = new TreeNode(7);
	root->right->right->right = new TreeNode(8);
	root->right->left->right->left = new TreeNode(9);
	root->right->right->right->right = new TreeNode(10);

	TreeNode *result = Solution::deepestLeftLeaf(root);
	if (result)
		cout << "The deepest left child is " << result->val << endl;
	else
		cout << "There is no left leaf in the given tree";
	
	
	
	vector<int> vec{ 10, 20, 15, 30, 20, 10, 10, 20, 9, 21 };
	PRINT(vec); SEP;
	merge_sort<int>(vec);
	PRINT(vec);

	for (int i = -16; i < 20; ++i) {
		PRINT(i);
		PRINT(isPowerOfTwo(i));
		SEP;
	}
	
	
	PRINT(hash1("hebing", 7));
	PRINT(hash2("hebing", 7));
	PRINT(hash3("hebing", 7));
	
	std::list<int> first;                                // empty list of ints
	std::list<int> second(4, 100);                       // four ints with value 100
	std::list<int> third(second.begin(), second.end());  // iterating through second
	std::list<int> fourth(third);                       // a copy of third

	// the iterator constructor can also be used to construct from arrays:
	int myints[] = { 16, 2, 77, 29 };
	std::list<int> fifth(myints, myints + sizeof(myints) / sizeof(int));

	std::cout << "The contents of fifth are: ";
	for (std::list<int>::iterator it = fifth.begin(); it != fifth.end(); it++)
		std::cout << *it << ' ';
	cout << endl;
	
	
	List<int>    theList;
	ListItr<int> theItr = theList.zeroth();
	printList(theList);

	for (int i = 0; i < 10; i++) {
		theList.insert(i, theItr);
		printList(theList);
		theItr.advance();
	}

	for (int i = 0; i < 10; i += 2)
		theList.remove(i);

	for (int i = 0; i < 10; i++)
		if ((i % 2 == 0) != (theList.find(i).isPastEnd()))
			cout << "Find fails!" << endl;

	cout << "Finished deletions" << endl;
	printList(theList);

	List<int> list2;
	list2 = theList;
	printList(list2);
	
	
	Vector<int> vec;
	vec.push_back(3); vec.push_back(4);
	PRINT(vec[0]);
	
	
	string s("bingtuo");
	cout << s << endl;
	s = "maomao";
	cout << s << endl;
	string s2("xiaobao");
	cout << (s < s2) << endl;
	cout << s.length() << endl;
	
	
	int ITEM_NOT_FOUND = -9999;
	HashTable<int> H(ITEM_NOT_FOUND);

	const int NUMS = 4000;
	const int GAP = 37;
	int i;

	cout << "Checking... (no more output means success)" << endl;

	for (i = GAP; i != 0; i = (i + GAP) % NUMS)
		H.insert(i);
	for (i = 1; i < NUMS; i += 2)
		H.remove(i);

	for (i = 2; i < NUMS; i += 2)
		if (H.find(i) != i)
			cout << "Find fails " << i << endl;

	for (i = 1; i < NUMS; i += 2) {
		if (H.find(i) != ITEM_NOT_FOUND)
			cout << "OOPS!!! " << i << endl;
	}
	
	matrix<int> m(3, 5);
	matrix<int> m2(m);
	PRINT(m[1]);
	PRINT(m.numrows());
	PRINT(m.numcols());
	
	
	Stack<int> s;

	for (int i = 0; i < 10; i++)
		s.push(i);

	while (!s.isEmpty())
		cout << s.topAndPop() << endl;
	
	
	Queue<int> q;

	for (int j = 0; j < 5; j++)
	{
		for (int i = 0; i < 5; i++)
			q.enqueue(i);

		while (!q.isEmpty())
			cout << q.dequeue() << endl;
	}

	q.enqueue(6); q.enqueue(7); q.enqueue(8);
	cout << q.isEmpty() << endl;
	cout << q.getFront() << endl;
	cout << q.dequeue() << endl;
	cout << q.dequeue() << endl;
	cout << q.dequeue() << endl;
	
	
	
	const int ITEM_NOT_FOUND = -9999;
	BinarySearchTree<int> t(ITEM_NOT_FOUND);
	int NUMS = 4000;
	const int GAP = 37;
	int i;

	cout << "Checking... (no more output means success)" << endl;

	for (i = GAP; i != 0; i = (i + GAP) % NUMS)
		t.insert(i);

	t.printTree();

	for (i = 1; i < NUMS; i += 2)
		t.remove(i);

	if (NUMS < 40)
		t.printTree();
	if (t.findMin() != 2 || t.findMax() != NUMS - 2)
		cout << "FindMin or FindMax error!" << endl;

	for (i = 2; i < NUMS; i += 2)
		if (t.find(i) != i)
			cout << "Find error1!" << endl;

	for (i = 1; i < NUMS; i += 2) {
		if (t.find(i) != ITEM_NOT_FOUND)
			cout << "Find error2!" << endl;
	}

	BinarySearchTree<int> t2(ITEM_NOT_FOUND);
	t2 = t;

	for (i = 2; i < NUMS; i += 2)
		if (t2.find(i) != i)
			cout << "Find error1!" << endl;

	for (i = 1; i < NUMS; i += 2) {
		if (t2.find(i) != ITEM_NOT_FOUND)
			cout << "Find error2!" << endl;
	}
	
	
	int ITEM_NOT_FOUND = -9999;
	HashTable<int> H(ITEM_NOT_FOUND);

	const int NUMS = 4000;
	const int GAP = 37;
	int i;

	cout << "Checking... (no more output means success)" << endl;

	for (i = GAP; i != 0; i = (i + GAP) % NUMS)
		H.insert(i);
	for (i = 1; i < NUMS; i += 2)
		H.remove(i);

	for (i = 2; i < NUMS; i += 2)
		if (H.find(i) != i)
			cout << "Find fails " << i << endl;

	for (i = 1; i < NUMS; i += 2)
	{
		if (H.find(i) != ITEM_NOT_FOUND)
			cout << "OOPS!!! " << i << endl;
	}
	
	
	
void checkSort(const vector<int> & a) {
	for (size_t i = 0; i < a.size(); i++)
		if (a[i] != i)
			cout << "Error at " << i << endl;
	cout << "Finished checksort" << endl;
}


void permute(vector<int> & a){
	static Random r;
	for (size_t j = 1; j < a.size(); j++)
		swap(a[j], a[r.randomInt(0, j)]);
}
	
	
	const int NUM_ITEMS = 1000;

	vector<int> a(NUM_ITEMS);
	for (size_t i = 0; i < a.size(); i++)
		a[i] = i;

	for (int theSeed = 0; theSeed < 20; theSeed++) {
		permute(a);
		insertionSort(a);
		checkSort(a);

		permute(a);
		heapsort(a);
		checkSort(a);

		permute(a);
		shellsort(a);
		checkSort(a);

		permute(a);
		mergeSort(a);
		checkSort(a);

		permute(a);
		quicksort(a);
		checkSort(a);

		permute(a);
		largeObjectSort(a);
		checkSort(a);

		permute(a);
		quickSelect(a, NUM_ITEMS / 2);
		cout << a[NUM_ITEMS / 2 - 1] << " " << NUM_ITEMS / 2 << endl;
	}
	
	int ITEM_NOT_FOUND = -9999;
	HashTable<int> H(ITEM_NOT_FOUND);

	const int NUMS = 4000;
	const int GAP = 37;
	int i;

	cout << "Checking... (no more output means success)" << endl;

	for (i = GAP; i != 0; i = (i + GAP) % NUMS) H.insert(i);
	
	for (i = 1; i < NUMS; i += 2) H.remove(i);

	for (i = 2; i < NUMS; i += 2)
		if (H.find(i) != i)
			cout << "Find fails " << i << endl;

	for (i = 1; i < NUMS; i += 2) {
		if (H.find(i) != ITEM_NOT_FOUND)
			cout << "OOPS!!! " << i << endl;
	}
	
	
	//int n = 10000000;
	//PRINT(pi_estimator(n));

	double xbar = PI / 4.;
	
	//PRINT(tmp);

	for (int i = 10; i < 1000000; i *= 10) {
		PRINT(i);
		PRINT(pi_estimator(i));
		double lower = 4 * xbar - 1.96 * 4 * sqrt(xbar*(1 - xbar) / i);
		double upper = 4 * xbar + 1.96 * 4 * sqrt(xbar*(1 - xbar) / i);
		vector<double> tmp{ lower, PI, upper };
		PRINT(tmp);
		SEP;
	}
	
	vector<int> nums{ 1, 4, 3, 2, 1 };
	PRINT(Solution::containsDuplicate(nums));
	PRINT(nums);
	nums.insert(nums.begin(), 8);
	PRINT(nums);
	
	
	vector<int> nums{ 9, 9, 9, 9, 9 };
	PRINT(Solution::plusOne(nums));
	string s("abc");
	s.insert(s.begin(), 'x');
	PRINT(s);
	
	vector<string> words{ "practice", "makes", "perfect", "coding", "makes" };
	string word1("coding"), word2("practice");
	PRINT(Solution::shortestDistance(words, word1, word2));
	
	
	vector<int> nums{ 1, 2, 3, 4, 5, 6, 7 };
	PRINT(nums);
	rotate(nums,3);
	SEP;
	PRINT(nums);
	
	vector<int> nums{ 1, 2, 3, 4, 5, 6, 7 };
	PRINT(removeElement(nums,4));
	PRINT(nums);
	PRINT(removeElement(nums, 5));
	PRINT(nums);
	PRINT(removeElement(nums, 2));
	PRINT(nums);
	
	
	map<char, int> mp;
	mp['a'] = 2;
	mp['b'] = 3;
	PRINT(mp['a']);
	PRINT(mp['b']);
	
	
	map<char, int> p2i; // map char to int
	map<string, int> w2i; // map string to int
	istringstream in(str); // parse the word strings
	int i = 0, n = pattern.size();
	for (string word; in >> word; ++i) {
		if (p2i[pattern[i]] != w2i[word] || i == n)
			return false; // if str is longer, or no match, return with false, before recording
		p2i[pattern[i]] = w2i[word] = i + 1; // record each char/string mapping
	}
	for (map<char, int>::iterator it = p2i.begin(); it != p2i.end(); ++it) {
		cout << *it << endl;
	}
	//cout << p2i << endl;
	SEP;
	cout << p2i << endl;
	SEP;
	cout << w2i << endl;
	return i == n;


	