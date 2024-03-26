% mass
m1 = 25000;
m2 = 20000;
m3 = 20000;
m4 = 18000;
m5 = 15000;

% stiffness
k1 = 5*1e6;
k2 = 4*1e6;
k3 = 4*1e6;
k4 = 3*1e6;
k5 = 3*1e6;

% M and K matrix
M5 = diag([m1,m2,m3,m4,m5]);
K5 = [k1+k2 -k2 0 0 0;...
    -k2 k2+k3 -k3 0 0;...
    0 -k3 k3+k4 -k4 0;...
    0 0 -k4 k4+k5 -k5;...
    0 0 0 -k5 k5];

% lets check natural frequency 
[phi, L] = eig(K,M);
omg = sqrt(diag(L));
natural_freq = omg/(2*pi);

damping_coef_diag = zeros(5,1);
for i =1:5
    pho = 0.01*i;
damping_coef_diag(i) = 2*pho*M(i,i)*omg(i);

end
damping_coef = diag(damping_coef_diag);
C5 = (phi')\damping_coef/phi;
save("5dof_MKC_matrix","M5","K5","C5")


