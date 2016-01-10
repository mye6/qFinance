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
	
	
	string pattern("abba"), str("dog cat cat dog");
	PRINT(Solution::wordPattern(pattern, str));


	vector<string> vec{ "deer", "door", "cake", "card" };
	ValidWordAbbr vwa(vec);
	PRINT(vwa.isUnique("deer"));
	PRINT(vwa.isUnique("de"));
	
	
	
	PRINT(count_lines());
	
	
	Integer a(2);
	PRINT(+a);
	PRINT(a);
	PRINT(a++);
	PRINT(++a);

	PRINT(a);
	Integer b(1);
	Integer c(2);
	a = b + c;
	PRINT(a);
	a += c;
	PRINT(a);
	a += a;
	PRINT(a);

	PRINT(count_lines());
	
	
	
	string str("bing mao love");
	istringstream is(str);
	string word;
	while (is >> word) cout << "#" << word << "#" <<endl; // get rid of the space in the middle
	
	
	// vector<int> nums{ 2, 4, 1, 5, 7, 0, 3, 8, 6, 9 };
	const int n = 11;
	int v[n] = { 2, 4, 1, 5, 7, 0, 3, 8, 6, 9, 10 };
	int k = n/2;
	PRINT(find_kth(v, n, k));
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	int numSquares(int n) {
	if (n <= 0) return 0;
	// D[i] = the least number of perfect square numbers 
	// which sum to i. Note that cntPerfectSquares[0] is 0.
	vector<int> D(n + 1, INT_MAX);
	D[0] = 0;
	for (int i = 1; i <= n; i++) {
		// For each i, it must be the sum of some number (i - j*j) and 
		// a perfect square number (j*j).
		for (int j = 1; j*j <= i; j++) {
			D[i] = min(D[i], D[i - j*j] + 1);
		}
	}
	return D[n];
}

int nthUglyNumber(int n) {
	if (n <= 0) return 0; // get rid of corner cases 
	if (n == 1) return 1; // base case
	int t2 = 0, t3 = 0, t5 = 0; //pointers for 2, 3, 5
	vector<int> k(n);
	k[0] = 1;
	for (int i = 1; i < n; i++) {
		k[i] = min(k[t2] * 2, min(k[t3] * 3, k[t5] * 5));
		if (k[i] == k[t2] * 2) t2++;
		if (k[i] == k[t3] * 3) t3++;
		if (k[i] == k[t5] * 5) t5++;
	}
	return k[n - 1];
}


int maxProfit(vector<int>& prices) {
	int n = prices.size();
	if (n < 2) return 0;

	int has1_doNothing = -prices[0];
	int has1_Sell = 0;
	int has0_doNothing = 0;
	int has0_Buy = -prices[0];	
	for (int i = 1; i<n; ++i) {
		has1_doNothing = max(has1_doNothing, has0_Buy);
		has0_Buy = -prices[i] + has0_doNothing;
		has0_doNothing = max(has0_doNothing, has1_Sell);
		has1_Sell = prices[i] + has1_doNothing;
	}
	return max(has1_Sell, has0_doNothing);
}


int minimumTotal(vector<vector<int>>& triangle) {
	vector<int> res(triangle.size(), triangle[0][0]);
	for (unsigned int i = 1; i < triangle.size(); i++)
		for (int j = i; j >= 0; j--) {
		if (j == 0)
			res[0] += triangle[i][j];
		else if (j == i)
			res[j] = triangle[i][j] + res[j - 1];
		else
			res[j] = triangle[i][j] + min(res[j - 1], res[j]);
		}
	return *min_element(res.begin(), res.end());
}

// There's a typical DP solution with O(N^2) Time and O(N) space 
// DP[i] means the result ends at i
// So for dp[i], dp[i] is max(dp[j]+1), for all j < i and nums[j] < nums[i]
int lengthOfLIS_1(vector<int>& nums) {
	const int size = nums.size();
	if (size == 0) { return 0; }
	vector<int> dp(size, 1);
	int res = 1;
	for (int i = 1; i < size; ++i) {
		for (int j = 0; j < i; ++j) {
			if (nums[j] < nums[i]) {
				dp[i] = max(dp[i], dp[j] + 1);
			}
		}
		res = max(res, dp[i]);
	}
	return res;
}

int minCost(vector<vector<int>>& costs) {
	int n = costs.size();
	for (int i = 1; i < n; i++) {
		costs[i][0] += std::min(costs[i - 1][1], costs[i - 1][2]);
		costs[i][1] += std::min(costs[i - 1][0], costs[i - 1][2]);
		costs[i][2] += std::min(costs[i - 1][0], costs[i - 1][1]);
	}
	return (n == 0) ? 0 : (std::min(costs[n - 1][0], std::min(costs[n - 1][1], costs[n - 1][2])));
}


int lengthOfLIS_2(vector<int>& nums) {
	vector<int> res;
	for (int i = 0; i<nums.size(); i++) {
		auto it = std::lower_bound(res.begin(), res.end(), nums[i]);
		if (it == res.end()) res.push_back(nums[i]);
		else *it = nums[i];
	}
	return res.size();
}

int minPathSum(vector<vector<int>>& grid) {
	int m = grid.size();
	int n = grid[0].size();
	vector<vector<int> > sum(m, vector<int>(n, grid[0][0]));
	for (int i = 1; i < m; i++)
		sum[i][0] = sum[i - 1][0] + grid[i][0];
	for (int j = 1; j < n; j++)
		sum[0][j] = sum[0][j - 1] + grid[0][j];
	for (int i = 1; i < m; i++)
		for (int j = 1; j < n; j++)
			sum[i][j] = min(sum[i - 1][j], sum[i][j - 1]) + grid[i][j];
	return sum[m - 1][n - 1];
}

int uniquePaths(int m, int n) {
	vector<vector<int> > path(m, vector<int>(n, 1));
	for (int i = 1; i < m; i++)
		for (int j = 1; j < n; j++)
			path[i][j] = path[i - 1][j] + path[i][j - 1];
	return path[m - 1][n - 1];
}


int maxProduct(vector<int>& A) {
	int n = A.size();
	if (n == 0) return 0;
	int maxProduct = A[0], minProduct = A[0], maxRes = A[0];
	for (int i = 1; i < n; i++) {
		if (A[i] >= 0) {
			maxProduct = max(maxProduct * A[i], A[i]);
			minProduct = min(minProduct * A[i], A[i]);
		}
		else {
			int temp = maxProduct;
			maxProduct = max(minProduct * A[i], A[i]);
			minProduct = min(temp * A[i], A[i]);
		}
		maxRes = max(maxRes, maxProduct);
	}
	return maxRes;
}

int maximalSquare(vector<vector<char>>& matrix) {
	int m = matrix.size();
	if (!m) return 0;
	int n = matrix[0].size();
	vector<vector<int> > size(m, vector<int>(n, 0));
	int maxsize = 0;
	for (int j = 0; j < n; j++) {
		size[0][j] = matrix[0][j] - '0';
		maxsize = max(maxsize, size[0][j]);
	}
	for (int i = 1; i < m; i++) {
		size[i][0] = matrix[i][0] - '0';
		maxsize = max(maxsize, size[i][0]);
	}
	for (int i = 1; i < m; i++) {
		for (int j = 1; j < n; j++) {
			if (matrix[i][j] == '1') {
				size[i][j] = min(size[i - 1][j - 1], min(size[i - 1][j], size[i][j - 1])) + 1;
				maxsize = max(maxsize, size[i][j]);
			}
		}
	}
	return maxsize * maxsize;
}


class Solution1 {
	bool DP_helper(unordered_map<string, bool>& isScramblePair, string s1, string s2) {
		int i, len = s1.size();
		bool res = false;
		if (len == 0) return true;
		else if (len == 1) return s1 == s2;
		
		if (isScramblePair.count(s1 + s2)) return isScramblePair[s1 + s2];
		// checked before, return intermediate result directly
		
		if (s1 == s2) res = true;
		for (i = 1; i<len && !res; ++i) {
			//check s1[0..i-1] with s2[0..i-1] and s1[i..len-1] and s2[i..len-1]
			res = res || (DP_helper(isScramblePair, s1.substr(0, i), s2.substr(0, i)) && DP_helper(isScramblePair, s1.substr(i, len - i), s2.substr(i, len - i)));
			//if no match, then check s1[0..i-1] with s2[len-k.. len-1] and s1[i..len-1] and s2[0..len-i]
			res = res || (DP_helper(isScramblePair, s1.substr(0, i), s2.substr(len - i, i)) && DP_helper(isScramblePair, s1.substr(i, len - i), s2.substr(0, len - i)));
		}		
		return isScramblePair[s1 + s2] = res; //save the intermediate results		
	}
public:
	bool isScramble(string s1, string s2) {
		unordered_map<string, bool> isScramblePair;
		return DP_helper(isScramblePair, s1, s2);
	}
};


class Solution {
public:
	bool isScramble(string s1, string s2) {		
		size_t n = s1.length();
		if (n <= 0) return false; // if s1 is empty, return true		
		
		// Dynamic Programming: 
		// eq[first1][first2][len] == true iff s1[first1 ... first1+len) == s2[first2 ... first2+len)
		vector<vector<vector<bool> > > eq
			(n, vector<vector<bool>>(n, vector<bool>(n + 1, false))); // initialize: all false

		// initialize: eq[first1][first2][1] = true iff s1[first1] == s2[first2]
		for (int first1 = 0; first1 < n; ++first1) {
			for (int first2 = 0; first2 < n; ++first2) {
				eq[first1][first2][1] = (s1[first1] == s2[first2]);
			}
		}

		// dp: eq[first1][first2][len] = true iff two substrings are both matched.
		for (size_t len = 2; len <= n; ++len) {
			for (size_t first1 = 0; first1 + len <= n; ++first1) {
				for (size_t first2 = 0; first2 + len <= n; ++first2) {
					for (size_t len1 = 1; len1 < len; ++len1) {
						size_t len2 = len - len1;

						// two substrings are not swapped
						if (eq[first1][first2][len1] && eq[first1 + len1][first2 + len1][len2]) {
							eq[first1][first2][len] = true;
						}

						// two substrings are swapped
						if (eq[first1][first2 + len2][len1] && eq[first1 + len1][first2][len2]) {
							eq[first1][first2][len] = true;
						}
					}
				}
			}
		}	
		
		return eq[0][0][n];
	}
};

int maxProfit(vector<int>& prices) {
	if (prices.empty()) return 0;
	int n = prices.size();
	vector<int> leftProfit(n), rightProfit(n);	
	int leftMin = prices[0], rightMax = prices[n - 1];
	leftProfit[0] = 0; rightProfit[n - 1] = 0;
	for (int i = 1, j = n-2; i <= n-1 && j >= 0; ++i, --j) {
		leftProfit[i] = max(leftProfit[i - 1], prices[i] - leftMin);
		leftMin = min(leftMin, prices[i]);
		rightProfit[j] = max(rightProfit[j + 1], rightMax - prices[j]);
		rightMax = max(rightMax, prices[j]);
	}
	
	int res = 0;
	for (int i = 1; i < n; ++i) {
		res = max(res, leftProfit[i] + rightProfit[i]);
	}
	return res;
}


*/

/*
int minCut(string s) {
	if (s.empty()) return 0;
	int n = s.size();
	vector<vector<bool>> pal(n, vector<bool>(n, false));
	vector<int> d(n);
	for (int i = n - 1; i >= 0; i--) {
		d[i] = n - i - 1;
		for (int j = i; j<n; j++) {
			if (s[i] == s[j] && (j - i<2 || pal[i + 1][j - 1])) {
				pal[i][j] = true;
				if (j == n - 1)
					d[i] = 0;
				else if (d[j + 1] + 1<d[i])
					d[i] = d[j + 1] + 1;
			}
		}
	}
	return d[0];
}

bool isMatch(string str, string pattern) {
	int s = 0, p = 0, match = 0, idx = -1, ns = str.size(), np = pattern.size();
	while (s < ns) {
		// advancing both pointers
		if (p < np && pattern[p] == '?' || str[s] == pattern[p]) {
			++s; ++p;
		}
		// * found, only advancing pattern pointer
		else if (p < np && pattern[p] == '*') {
			idx = p;
			match = s;			
			++p;
		}
		// last pattern pointer was *, advancing string pointer
		else if (idx != -1) {
			p = idx + 1;
			++match;
			s = match;
		}
		//current pattern pointer is not star, last patter pointer was not *
		//characters do not match
		else 
			return false;
	}
	//check for remaining characters in pattern
	while (p < pattern.length() && pattern[p] == '*')
		p++;
	
	return p == np;
}


int maximalRectangle(vector<vector<char> > &matrix) {
	if (matrix.empty()) return 0;
	const int m = matrix.size();
	const int n = matrix[0].size();
	vector<int> left(n, 0), right(n, n), height(n, 0);
	int maxA = 0;
	for (int i = 0; i<m; i++) {
		PRINT(i);
		int cur_left = 0, cur_right = n;
		for (int j = 0; j<n; j++) { // compute height (can do this from either side)
			if (matrix[i][j] == '1') height[j]++;
			else height[j] = 0;
		}
		for (int j = 0; j<n; j++) { // compute left (from left to right)
			if (matrix[i][j] == '1') left[j] = max(left[j], cur_left);
			else { left[j] = 0; cur_left = j + 1; }
		}
		// compute right (from right to left)
		for (int j = n - 1; j >= 0; j--) {
			if (matrix[i][j] == '1') right[j] = min(right[j], cur_right);
			else { right[j] = n; cur_right = j; }
		}
		// compute the area of rectangle (can do this from either side)
		for (int j = 0; j<n; j++)
			maxA = max(maxA, (right[j] - left[j])*height[j]);
		PRINT(height);
		PRINT(left);
		PRINT(right);

		SEP;
	}
	return maxA;
}

int longestValidParentheses(string s) {
	stack<int> stk;
	stk.push(-1);
	int maxL = 0;
	for (int i = 0; i<s.size(); i++) {
		int t = stk.top();
		if (t != -1 && s[i] == ')' && s[t] == '(') {
			stk.pop();
			maxL = max(maxL, i - stk.top());
		}
		else
			stk.push(i);
	}
	return maxL;
}


class Solution3 {
public:
	bool isMatch(string s, string p) {
		if (p.empty())    return s.empty();

		if ('*' == p[1])
			// x* matches empty string or at least one character: x* -> xx*
			// *s is to ensure s is non-empty
			return (isMatch(s, p.substr(2)) || !s.empty() && (s[0] == p[0] || '.' == p[0]) && isMatch(s.substr(1), p));
		else
			return !s.empty() && (s[0] == p[0] || '.' == p[0]) && isMatch(s.substr(1), p.substr(1));
	}
};

class Solution4 {
public:
	bool isMatch(string s, string p) {
		/**
		* f[i][j]: if s[0..i-1] matches p[0..j-1]
		* if p[j - 1] != '*'
		*      f[i][j] = f[i - 1][j - 1] && s[i - 1] == p[j - 1]
		* if p[j - 1] == '*', denote p[j - 2] with x
		*      f[i][j] is true iff any of the following is true
		*      1) "x*" repeats 0 time and matches empty: f[i][j - 2]
		*      2) "x*" repeats >= 1 times and matches "x*x": s[i - 1] == x && f[i - 1][j]
		* '.' matches any single character
		*/
		int m = s.size(), n = p.size();
		vector<vector<bool>> f(m + 1, vector<bool>(n + 1, false));

		f[0][0] = true;
		for (int i = 1; i <= m; i++)
			f[i][0] = false;
		// p[0.., j - 3, j - 2, j - 1] matches empty iff p[j - 1] is '*' and p[0..j - 3] matches empty
		for (int j = 1; j <= n; j++)
			f[0][j] = j > 1 && '*' == p[j - 1] && f[0][j - 2];

		for (int i = 1; i <= m; i++)
			for (int j = 1; j <= n; j++)
				if (p[j - 1] != '*')
					f[i][j] = f[i - 1][j - 1] && (s[i - 1] == p[j - 1] || '.' == p[j - 1]);
				else
					// p[0] cannot be '*' so no need to check "j > 1" here
					f[i][j] = f[i][j - 2] || (s[i - 1] == p[j - 2] || '.' == p[j - 2]) && f[i - 1][j];

		return f[m][n];
	}
};

// Hash Table
class TwoSum {
	unordered_map<int, int> map;
public:
	void add(int number) {
		map[number]++;
	}

	bool find(int value) {
		for (unordered_map<int, int>::iterator it = map.begin(); it != map.end(); it++) {
			int i = it->first;
			int j = value - i;
			if ((i == j && it->second > 1) || (i != j && map.find(j) != map.end())) {
				return true;
			}
		}
		return false;
	}
};

// only contains digits 
string getHint(string secret, string guess) {
	int aCnt = 0;
	int bCnt = 0;
	vector<int> sVec(10, 0); // 0 ~ 9 for secret
	vector<int> gVec(10, 0); // 0 ~ 9 for guess 
	if (secret.size() != guess.size() || secret.empty()) { return "0A0B"; }
	for (int i = 0; i < secret.size(); ++i) {
		char c1 = secret[i]; char c2 = guess[i];
		if (c1 == c2) {
			++aCnt;
		}
		else {
			++sVec[c1 - '0'];
			++gVec[c2 - '0'];
		}
	}
	// count b 
	for (int i = 0; i < sVec.size(); ++i) {
		bCnt += min(sVec[i], gVec[i]);
	}
	return to_string(aCnt) + 'A' + to_string(bCnt) + 'B';
}

bool containsNearbyDuplicate(vector<int>& nums, int k) {
	unordered_map<int, int> hashMap;
	for (int i = 0; i < nums.size(); ++i) {
		if (hashMap.find(nums[i]) != hashMap.end() && i - hashMap[nums[i]] <= k)  return true;
		hashMap[nums[i]] = i;
	}
	return false;
}

int next(int n) {
	int res = 0;
	while (n) {
		int t = n % 10;
		res += t*t;
		n /= 10;
	}
	return res;
}

bool isHappy(int n) {
	int i1 = n, i2 = next(n);
	while (i2 != i1) {
		i1 = next(i1);
		i2 = next(next(i2));
	}
	return i1 == 1;
}


Queue q;
	q.push(1);
	q.push(2);
	q.push(3);
	q.push(4);
	q.push(5);

	for (int i = 0; i < 5; ++i) {
		PRINT(q.peek()); q.pop();
	}
	
	
	
	vector<int> data{ 1, 3, 5, 2, -1, 9 };
	int n = data.size();
	MinStack ms;
	for (int i = 0; i < n; ++i) {
		cout << i << endl;
		ms.push(data[i]);
		PRINT(ms.getMin());
		PRINT(ms.top());
		SEP;
	}
	SEP;
	SEP;
	for (int i = 0; i < n; ++i) {
		cout << i << endl;
		PRINT(ms.top());
		PRINT(ms.getMin());
		ms.pop();
		SEP;
	}
	
	
	Stack st;
	st.push(1);
	st.push(2);
	st.push(3);
	st.push(4);
	st.push(5);

	for (int i = 0; i < 5; ++i) {
		PRINT(st.top()); st.pop();
	}
	
	
	vector<int>	nums{ 1, 3, 5, 7, 9 };
	ListNode* head = genList(nums);
	cout << head << endl;
	head = removeNthFromEnd(head, 2);
	cout << head << endl;
	ListNode* rev = reverseList(head);
	cout << rev << endl;	
	clear(rev);
	
	
	vector<int>	nums1{ 1, 3, 5, 7, 9 };
	vector<int>	nums2{ -2, 3, 4, 8, 10 };
	ListNode *l1 = genList(nums1), *l2 = genList(nums2);
	PRINT(l1);
	PRINT(l2);

	l1 = mergeTwoLists(l1, l2);
	PRINT(l1);
	
	vector<int>	nums2{ -2, 3, 4, 8, 10 };
	ListNode *l1 = genList(nums1), *l2 = genList(nums2);
	PRINT(l1);
	PRINT(l2);

	l1 = mergeTwoLists(l1, l2);
	PRINT(l1);
	PRINT(countNodes(l1));
	
	
	vector<int>	nums{ 1, 1, 2 };
	ListNode* head = genList(nums);
	PRINT(head);
	head = removeElements(head, 2);
	PRINT(head);
	
	vector<int>	nums1{1, 2, 3};
	vector<int>	nums2{ 4, 5, 6, 7 };
	vector<int>	nums3{ 8, 9, 10, 11, 12 };
	ListNode *head1 = genList(nums1);
	ListNode *head2 = genList(nums2);
	ListNode *head3 = genList(nums3);
	ListNode *tail1 = findTail(head1);
	ListNode *tail2 = findTail(head2);
	tail1->next = head3;
	tail2->next = head3;
	PRINT(head1);
	PRINT(head2);
	ListNode* x = getIntersectionNode(head1, head2);
	PRINT(x->val);

	PRINT(count_lines());
	
	
	
	
	
	vector<int> nums{ 1, 2, 3, 4, 5, 6, 7 };
	TreeNode* t = sortedArrayToBST(nums);
	cout << t << endl;
	
	
	
	vector<int> nums{ 1, 2, 3, 4, 5, 6, 7 };
	TreeNode* root = sortedArrayToBST(nums);
	cout << root << endl;
	Codec cod;
	cout << cod.serialize(root) << endl;
	TreeNode* to = cod.deserialize(cod.serialize(root));
	cout << to << endl;
	
	string s("1 2 # # 3 4 # # 5 # #");
	TreeNode* to2 = cod.deserialize(s);
	cout << to2 << endl;

	vector<int> vec = inorderTraversal(root);
	PRINT(vec);
	vector<int> vec2 = inorderTraversal(to2);
	PRINT(vec2);

	vector<int> vec3 = preorderTraversal(to2);
	PRINT(vec3);

	vector<int> vec4 = postorderTraversal(to2);
	PRINT(vec4);
	
	cout << to2 << endl;
	PRINT(maxDepth2(to2));
	
	
	
	
	Codec cod;
	//string s("1 2 # # 3 4 # # 5 # #");
	string s("3 9 # # 20 15 # # 7 # #");
	TreeNode* root = cod.deserialize(s);
	cout << root << endl;
	PRINT(closestValue(root, 3.));
	PRINT(zigzagLevelOrder(root));
	PRINT(rightSideView(root));
	PRINT(hasPathSum(root, 29));
	
	
	
	Codec cod;
	string s("6 2 0 # # 4 3 # # 5 # # 8 7 # # 9 # #");
	TreeNode* t = cod.deserialize(s);
	vector<string> vec = binaryTreePaths(t);
	cout << vec << endl;
	
	PRINT(numTrees(3));
	auto trees = generateTrees(3);
	for (auto tree : trees) {
		SEP;
		cout << tree << endl;
	}
	
	
	
	Codec cod;
	string s("6 2 0 # # 4 3 # # 5 # # 8 7 # # 9 # #");
	TreeNode* t = cod.deserialize(s);
	flatten(t);
	cout << t << endl;
	
	
	Codec cod;
	string s("6 2 0 # # 4 3 # # 5 # # 8 7 # # 9 # #");
	TreeNode* t = cod.deserialize(s);
	cout << t << endl;
	SEP;
	BSTIterator bsti(t);
	
	while (bsti.hasNext()) {
		PRINT(bsti.next());
	}
	
	
	
	vector<int> nums{ 3, 1, 5, 1 };
	TwoSum ts;
	for (int i : nums) ts.add(i);
	
	
	
	
	int n = 4;
	int i1 = 20, i2 = digitSquareSum(i1);
	while (i1 != i2 || --n>0) {
		cout << i1 << " " << i2 << endl;
		i1 = digitSquareSum(i1);
		i2 = digitSquareSum(digitSquareSum(i2));
	}
	
	
	
	
	
	
	vector<string> strs{ "eat", "tea", "tan", "ate", "nat", "bat" };
	PRINT(groupAnagrams(strs));
	
	string a = "1"; string b = "101";
	string res = addBinary0(a, b);
	PRINT(res);
	
	
	
	
	
	vector<int> nums1{ 2, 4, 3 }, nums2{ 5, 6, 4 };
	ListNode* l1 = genList(nums1);
	ListNode* l2 = genList(nums2);
	ListNode* res = addTwoNumbers(l1, l2);
	PRINT(l1);
	PRINT(l2);
	PRINT(res);
	
	
	
	PRINT(trailingZeroes(100));
	
	
	for (int i = 1; i <= 28; ++i) {
		cout << i << ": " << convertToTitle(i) << endl;
	}
	
	
	for (int i = 1; i <= 99; ++i) {
		cout << i << ": " << numberToWords(i) << endl;
	}
	
	
	for (int i = 1; i <= 50; ++i) {
		cout << i << ": " << Fib1(i) << ", " << Fib1(i)
			<< ", " << Fib2(i) << ", " << Fib3(i) 
			<< ", " << Fib4(i) <<  endl;		
	}
	
	
	for (int i = 1; i <= 30; ++i) {
		cout << i << ": " << Fib1(i) << ", " << Fib2(i) << ", " <<  Fib3(i) << endl;
	}
	SEP;
	for (int i = 30; i >= 1; --i) {
		cout << i << ": " << Fib1(i) << ", " << Fib2(i) << ", " << Fib3(i) << endl;
	}
	
	
	for (int i = 0; i <= 32; ++i) {
		PRINT(i);
		PRINT(isPowerOfTwo1(i));
		PRINT(isPowerOfTwo2(i));
		SEP;
	}

	
	
	vector<int> nums{ 1, 1, 1, 2, 2, 3 };
	ListNode* l = genList(nums);
	l = deleteDuplicates(l);
	PRINT(l);
	
	PRINT(majorityElement1(nums));
	PRINT(majorityElement2(nums));
	PRINT(majorityElement3(nums));
	PRINT(majorityElement4(nums));
	
	
	
	
	vector<int> nums{ 1, 5, 1, 1, 1, 9, 3, 4, 3, 2 };
	PRINT(majorityElementII(nums));
	
	
	vector<int> nums{ 3, 5, 2, 1, 6, 4 };
	PRINT(nums);
	wiggleSort(nums);
	PRINT(nums);
	
	PRINT(count_lines());
	
	
	vector<int> nums{ 2, 3, 6, 7 };
	PRINT(combinationSum2(nums, 7));
	
	
	
	

	