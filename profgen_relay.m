%function [dc,rho,dv,lambda,newRate,thres,EbN0mindB] = profgen_relay(R,t,Ps_bc,Ps_mac,Pr_mac,distance_sr,deg_rho11,rho11,deg_lambda11,lambda11,deg_rho2_min,deg_lambda_max);
clear all;
close all;
warning off MATLAB:fzero:UndeterminedSyntax;

tic
TRUE = 1;
FALSE = 0;
DEBUG_FLAG = 0;
f1 = inline('Phi-(1-3/x)*sqrt(pi/x)*exp(-x/4)','x','Phi');
f2 = inline('(1+1/(7*x))*sqrt(pi/x)*exp(-x/4)-Phi','x','Phi');


SNR = 0 % in dB

switch SNR;

case -5; % xcorr = 1
R = 0.25005/log(2); % 0.3607
t = 0.61;
Ps_bc =  1.082;
Ps_mac = 0.17436;
Pr_mac = 0.69744;
distance_sr = .5;
deg_rho11 = [10];rho11 = [1];deg_lambda11 = [2,3,6,7,15];lambda11 = [0.2288900000,0.2339420000,0.2163900000,0.0016320400,0.3191470000]; % rate = 0.6 (0.5914)
deg_rho2_min = 8;
deg_lambda_max = 25;
thresh_th = 0;

case -4; % xcorr = 1
R = 0.28783/log(2);
t = 0.61;
Ps_bc =  1.1148;
Ps_mac = 0.1641;
Pr_mac = 0.65641;
distance_sr = .5;
deg_rho11 = [12];rho11 = [1];deg_lambda11 = [2,3,5,7,12];lambda11 = [0.2275520000,0.2575070000,0.0892804000,0.1256160000,0.3000440000]; % rate = 0.68
%deg_rho11 = [17,18];rho11 = [0.5,0.5];deg_lambda11 = [2,3,13,14,55,56,58,66,67,68,69];lambda11 = [0.0571895000,0.3504700000,0.1964460000,0.1173770000,0.0651924000,0.0101712000,0.0089600000,0.0392549000,0.0608056000,0.0014702300,0.0926629000]; % rate = 0.67
%deg_rho11 = [15];rho11 = [1];deg_lambda11 = [2,3,4,5,6,20];lambda11 = [0.1736540000,0.1369820000,0.1497250000,0.0340162000,0.0001461340,0.5054770000]; % rate = 0.67
%deg_rho11 = [11];rho11 = [1];deg_lambda11 = [2,3,8];lambda11 = [0.2470300000,0.2776610000,0.4753100000]; % rate = 0.67
%deg_rho11 = [10];rho11 = [1];deg_lambda11 = [2,3,7,8];lambda11 = [0.2787440000,0.3194230000,0.3894640000,0.0123685000]; % rate = 0.67
%deg_rho11 = [9];rho11 = [1];deg_lambda11 = [2,3,5,6];lambda11 = [0.3245880000,0.3091480000,0.3093900000,0.0568743000]; % rate = 0.68
deg_rho2_min = [6];
deg_lambda_max = 25;
thresh_th = 1.50107198613575;

case -3; % xcorr = 1
R = 0.32684/log(2); % (=0.4715)
t = 0.62;
Ps_bc =  1.129;
Ps_mac = 0.15789;
Pr_mac = 0.63158;
distance_sr = .5;
deg_rho11 = [14];rho11 = [1];deg_lambda11 = [2,3,7];lambda11 = [0.2440670000,0.2923750000,0.4635580000]; % rate = 0.75 (0.7619)
deg_rho2_min = 8;
deg_lambda_max = 25;
thresh_th = 1.32939190230362;

case -2; % xcorr = 1
R = 0.5278; %0.36652/log(2);
t = 0.6284; %0.64;
Ps_bc =  1.1724; %1.1094;
Ps_mac = 0.1829; %0.16111;
Pr_mac = 0.5256; %0.64444;
distance_sr = .5;
deg_rho11 = [19,20];rho11 = [0.5,0.5];deg_lambda11 = [2,3,7,8];lambda11 = [0.2284350000,0.3363080000,0.2441790000,0.1910780000]; % rate = 0.82
deg_rho2_min = 7;rho2local = [.5,.5];
deg_lambda_max = 100;
thresh_th = 1.2589; %1.19524220275926;

case -1; % xcorr = 1
R = 0.40562/log(2);
t = 0.65;
Ps_bc =  1.1846;
Ps_mac = 0.12486;
Pr_mac = 0.53229;
distance_sr = .5;
deg_rho11 = [30];rho11 = [1];deg_lambda11 = [2,3,4,5];lambda11 = [0.2643850000,0.3178970000,0.2326350000,0.1850840000]; % rate = 0.9 (0.9003)
deg_rho2_min = 8;
deg_lambda_max = 35;
thresh_th = 1.03089429408649;


case 0; % xcorr = 1
R = 0.6410; %0.44375/log(2);
t = 0.6835; %0.68;
Ps_bc =  1.1476; %1.1765;
Ps_mac = 0.1432; %0.08125;
Pr_mac = 0.5381; %0.54375;
distance_sr = .5;
deg_rho11 = [46,47];rho11 = [0.125,0.875];deg_lambda11 = [2,3,4];lambda11 = [0.2774490000,0.4345360000,0.2880150000]; % rate = 0.9415
deg_rho2_min = 7;rho2local = [.125,.875];
deg_lambda_max = 10;
thresh_th = 1; %0.92194292151480;

case 1; % xcorr = 1
R = 0.48079/log(2);
t = 0.72;
Ps_bc =  1.1389;
Ps_mac = 0.12857;
Pr_mac = 0.51429;
distance_sr = 0.5;
deg_rho11 = [100];rho11 = [1];deg_lambda11 = [2,3,4,5];lambda11 = [0.2339440000,0.3214470000,0.4058120000,0.0387971000]; % rate = 0.97 (0.9670)
deg_rho2_min;
deg_lambda_max;
thresh_th = 0;


case 2; % xcorr = 1
R = 0.7470;
t = 0.7573;
Ps_bc =  1.1423;
Ps_mac = 0.1196;
Pr_mac = 0.4366;
distance_sr = 0.5;
deg_rho11 = [100];rho11 = [1];deg_lambda11 = [2,3,4,5];lambda11 = [0.2339440000,0.3214470000,0.4058120000,0.0387971000]; % rate = 0.97 (0.9856)
deg_rho2_min;
deg_lambda_max;
thresh_th = 0.7943;

end;


% ---------- Initialization -----------------------

if ~exist('deg_lambda_max','var');
    deg_lambda_max = max(deg_lambda11);
end;
R_bc = 1-sum(rho11./deg_rho11)/sum(lambda11./deg_lambda11);
R_mac_ub = 1-(1-R-t*(1-R_bc));
R_mac_lb = 1-(1-R-t*(1-R_bc))/(1-t);
	if DEBUG_FLAG;
	disp([num2str(R_mac_lb),' < Rmac < ',num2str(R_mac_ub)]);
	end;
P_bc = Ps_bc;			% at the final destination (distance source-destination = 1)
P_mac = Pr_mac/(1-distance_sr)^2 + Ps_mac;
tc = (1-R_bc)*t/(1-R);

dv11 = [min(deg_lambda11):max(deg_lambda11)];
[void, IA, IB] = intersect(dv11,deg_lambda11);tmp = zeros(size(dv11));tmp(IA) = lambda11;lambda11 = tmp;
lambda11p = lambda11./dv11;lambda11p = lambda11p/sum(lambda11p);


end_while = FALSE;


dv1 = [2:deg_lambda_max];
dv2 = dv1;

Asub = [toeplitz(ones(length(dv1),1),[1,zeros(1,length(dv1)-1)])*diag(1./dv1) , zeros(length(dv1),length(dv2)) ];
bsub = toeplitz(ones(length(dv1),1),[1,zeros(1,length(dv11)-1)])*reshape(lambda11p,length(lambda11p),1);

Aeq = [ones(1,length(dv1)+length(dv2));[1/t*1./dv1,-1/(1-t)*1./dv2]];
beq = [1;0];

lambda1 = zeros(size(dv1));
lambda2 = zeros(size(dv2));

dc1 = deg_rho11;
dc2 = [deg_rho2_min,deg_rho2_min+1];       % due to the concentration theorem
rho11p = rho11./deg_rho11;rho11p = rho11p/sum(rho11p);
rho1p = tc*rho11p;

C = [1./dv1,1./dv2];
d = 1/min(min(dc1),min(dc2))/.001; % max rate = 0.999

sigma_opt_dc = [0,Inf];
step = 0.1;

lambda_1 = (1e-2)*ones(length(lambda1)+length(lambda2),1);






%disp(' ---------- Optimization---------------------');


rho2plocal = rho2local./dc2;rho2plocal = rho2plocal/sum(rho2plocal);
rho2p = (1-tc)*rho2plocal;
rho = [rho1p,rho2p].*[dc1,dc2];rho = rho/sum(rho);
rho1 = rho(1:length(dc1));
rho2 = rho(length(dc1)+1:length(rho));

sigma_min = .2;
sigma_max = 1.17/R;
sigma = sigma_min;
newRate = Inf;
newRate_1 = 0;
RateUB = 0.8;
norm_diff_lambda = Inf;
min_norm_diff = 0.1;
iter = 0;
lock_flag = 0;
feasiblesolution_flag = 0;
conv_thres_lambda = 2e-2;
conv_thres_rate = 1e-4;
maxiter = 20;
while ( ((abs(R-newRate) > conv_thres_rate) & iter<maxiter) | feasiblesolution_flag == 0);
		%if DEBUG_FLAG;
		absR_newRate = abs(R-newRate)
		%end;
	iter = iter+1

	% --- Initialization for optimization
	s1 = 2*P_bc/sigma^2;
	s2 = 2*P_mac/sigma^2;
	%lambda21max = min(1-eps,exp(P_bc/2/sigma^2)/sum([rho1,rho2].*[dc1,dc2]))
	%lambda22max = min(1-eps,exp(P_mac/2/sigma^2)/sum([rho1,rho2].*[dc1,dc2]))
	lambda21max = max([0,((sum(rho2./dc2)+sum(rho1./dc1))/(1-R)-1/3)/(1/2-1/3),min([((sum(rho1./dc1)+sum(rho2./dc2))/(1-R)-1/sum(dv1(length(dv1))+dv2(length(dv2))))/(1/2-1/sum(dv1(length(dv1))+dv2(length(dv2)))),exp(1/2/sigma^2)/(sum(rho1.*dc1)+sum(rho2.*dc2)),1])]);lambda22max = lambda21max;
	%lambdaof2max = .2;	% To improve
        if norm_diff_lambda < conv_thres_lambda;
		maxr = sum(lambda(1:length(dv1)))*phi(s1)+sum(lambda(length(dv1)+1:length(dv1)+length(dv2)))*phi(s2);
	else;
		maxr = phi(min([s1,s2]));
	end;
	r1 = linspace(eps,maxr,100);
	r2 = linspace(eps,maxr,100);
	r1 = reshape(repmat(reshape(r1,length(r1),1),1,length(r2)),1,length(r1)*length(r2));
	r2 = reshape(repmat(reshape(r2,1,length(r2)),length(r2),1),1,length(r2)*length(r2));
	fr_tmp1 = zeros(size(r1));
	fr_tmp2 = zeros(size(r1));
	A = zeros(length(r1),length(dv1)+length(dv2));
	for n = 1:length(r1);
		r1n = r1(n);r2n = r2(n);
		tmp1 = 0;tmp2 = 0;
		for jc = 1:length(dc1);
			tmp1 = tmp1+rho1(jc)*phi_1(1-(1-r1n)^(dc1(jc)-1),f1,f2);
		end;
		for jc = 1:length(dc2);
			tmp2 = tmp2+rho2(jc)*phi_1(1-(1-r1n-r2n)^(dc2(jc)-1),f1,f2);
		end;
		tmp1 = tmp1 + tmp2;
		fr_tmp1(n) = tmp1;
		fr_tmp2(n) = tmp2;
	end;
	for lr = 1:length(r1);
		for iv = 1:length(dv1);
			A(lr,iv) = phi(s1+(dv1(iv)-1)*fr_tmp1(lr));
		end;
		for iv = 1:length(dv2);
			A(lr,length(dv1)+iv) = phi(s2+(dv2(iv)-1)*fr_tmp2(lr));
		end;
	end;
	if norm_diff_lambda < conv_thres_lambda;
		disp('locked with lambda');
		lock_flag = 1;
		alpha = sum(lambda(1:length(dv1))./dv1');
	%else
	%if abs(newRate_1-newRate) < conv_thres_rate;
	%	disp('locked with Rate');
	%	lock_flag = 0;
	%	alpha = t*(sum(rho1./dc1)+sum(rho2./dc2))/(1-newRate);
	else;
		disp('unlocked (norm lambda or rate)');
		lock_flag = 0;min_norm_diff = min_norm_diff*2;
		alpha = t*(sum(rho1./dc1)+sum(rho2./dc2))/(1-RateUB);
	end;
	A = [A;[1,zeros(1,length(dv1)-1),1,zeros(1,length(dv2)-1)];1/alpha*Asub;[-1./dv1,-1./dv2]];
	b = [reshape(r1+r2,length(r1),1);lambda21max+lambda22max;bsub;-(sum(rho1./dc1)+sum(rho2./dc2))];
	Aeq = [[ones(1,length(dv1)),ones(1,length(dv2))];[1/t*1./dv1,-1/(1-t)*1./dv2]];
	beq = [1;0];
	LB = zeros(length(dv1)+length(dv2),1);
	UB = [lambda21max;ones(length(dv1)-1,1);lambda22max;ones(length(dv2)-1,1)];
	
	% --- Optimization
	lambda=lsqlin(C,d,A,b,Aeq,beq,LB,UB);

	% --- Test if feasible solution
	alpha_opt = sum(lambda(1:length(dv1))./dv1');
	b_Ax = bsub - (1/alpha_opt)*Asub*lambda;
	if length(find(b_Ax<-(1e-4)))~=0;
		disp('Not a feasible solution');maxiter = maxiter+1;feasiblesolution_flag = 0;
	else;
		disp('Feasible solution');feasiblesolution_flag = 1;
	end;
		
		%if DEBUG_FLAG;
		matdisp(R,R_bc,t,lambda11p',dv11,lambda,dv1,dv2);pause(1); % Only for the DEMO
		%end;

	% --- Updating threshold
        norm_diff_lambda = norm(lambda-lambda_1)
        lambda_1 = lambda;
	newRate_1 = newRate;
        newRate = 1-(sum(rho1./dc1)+sum(rho2./dc2))/sum(lambda./([dv1,dv2])');
	if (lock_flag == 1) & (abs(newRate_1-newRate) < 1e-2);RateUB = newRate;end;
        if newRate < R;
        	sigma_max = sigma;
                sigma = (sigma_min+sigma)/2
        else;
                sigma_min = sigma;
                sigma = (sigma_max+sigma)/2
        end;
end;

thres = sigma;
EbN0mindB = 10*log10(1/(2*newRate*thres^2));

%disp('-------- End of optimization ----------');








% ---------------------- Termination ------------------------------------


% ------- Normalized Profile Determination for H -----------

rho_H = [rho1,rho2];
deg_rho_H = [dc1,dc2];
rhop_H = rho_H./deg_rho_H;rhop_H = rhop_H/sum(rhop_H);
lambda_H = lambda(1:length(dv1))+lambda(length(dv1)+1:length(dv1)+length(dv2));lambda_H = reshape(lambda_H,1,length(lambda_H)); %% Only if length(dv1) = length(dv2)
deg_lambda_H = [dv1];
lambdap_H = reshape(lambda_H,length(lambda_H),1)./reshape(deg_lambda_H,length(deg_lambda_H),1);lambdap_H = lambdap_H/sum(lambdap_H);
R_bis = 1-sum(rho_H./deg_rho_H)/sum(lambda_H./deg_lambda_H);

	if DEBUG_FLAG;
	nb1s_row_H = (1-R_bis)*sum(rhop_H.*deg_rho_H)
	nb1s_col_H = sum(lambdap_H.*deg_lambda_H')
	end;

% ------- Normalized Profile Determination for H11 -----------

rho_H11 = rho11;
deg_rho_H11 = deg_rho11;
rhop_H11 = rho_H11./deg_rho_H11;rhop_H11 = rhop_H11/sum(rhop_H11);
lambda_H11 = lambda11;
deg_lambda_H11 = dv11;
lambdap_H11 = reshape(lambda_H11,length(lambda_H11),1)./reshape(deg_lambda_H11,length(deg_lambda_H11),1);lambdap_H11 = lambdap_H11/sum(lambdap_H11);
R11_bis = 1-sum(rho_H11./deg_rho_H11)/sum(lambda_H11./deg_lambda_H11);

	if DEBUG_FLAG;
	nb1s_row_H11 = t*(1-R11_bis)*sum(rhop_H11.*deg_rho_H11)
	nb1s_col_H11 = t*sum(lambdap_H11.*deg_lambda_H11')
	end;


% ------- Normalized Profile Determination for H2 -----------

rho_H2 = rho2/sum(rho2);
deg_rho_H2 = dc2;
rhop_H2 = rho_H2./deg_rho_H2;rhop_H2 = rhop_H2/sum(rhop_H2);

deg_lambda_H22 = dv2;
lambdap_H22 = lambda(length(dv1)+1:length(dv1)+length(dv2))./(dv2');lambdap_H22 = lambdap_H22/sum(lambdap_H22);

lambdap11 = lambdap_H11/sum(lambdap_H11);
cs_lambdap11 = cumsum(lambdap11);
lambdap1 = (lambda(1:length(dv1))./dv1');lambdap1 = lambdap1/sum(lambdap1);
deg_lambda1 = dv1;
cs_lambdap1 = cumsum(lambdap1);
cs_lambdap21 = sort(unique([cs_lambdap1;cs_lambdap11]));
lambdap_H21 = [cs_lambdap21(1);cs_lambdap21(2:end)-cs_lambdap21(1:end-1)];
deg_lambda_H21 = zeros(size(lambdap_H21));for n = 1:length(deg_lambda_H21);deg_lambda_H21(n) = dv1(min(find(cs_lambdap1>=cs_lambdap21(n)-(1e-6)))) - deg_lambda_H11(min(find(cs_lambdap11>=cs_lambdap21(n)-(1e-6))));end;

	%if DEBUG_FLAG;
	AA = [toeplitz(ones(length(dv1),1),[1,zeros(1,length(dv1)-1)])];
	bb = toeplitz(ones(length(dv1),1),[1,zeros(1,length(dv11)-1)])*reshape(lambdap11,length(lambdap11),1);
	b_Ax_opt = bb - AA*lambdap1;
	if length(find(b_Ax_opt<-(1e-4))) ~= 0;disp('Not a feasible solution');end;
	%matdisp(R,R_bc,t,lambda11p',dv11,lambda,dv1,dv2);
	%end;

deg_lambda_H2 = [reshape(deg_lambda_H21,1,length(deg_lambda_H21)),reshape(deg_lambda_H22,1,length(deg_lambda_H22))];
lambdap_H2 = [t*reshape(lambdap_H21,1,length(lambdap_H21)),(1-t)*reshape(lambdap_H22,1,length(lambdap_H22))];
R2_bis = 1-(1-R)*(1-tc);

	%if DEBUG_FLAG;
	nb1s_row_H2 = (1-R-t*(1-R11_bis))*sum(rhop_H2.*deg_rho_H2)
	nb1s_col_H2 = sum(lambdap_H2.*deg_lambda_H2)
	%end;


% --------- Zeros removal -----------------------------------------


inonzeros = find(abs(lambdap_H11)>1e-6);deg_lambda_H11 = deg_lambda_H11(inonzeros);lambdap_H11 = lambdap_H11(inonzeros);
inonzeros = find(abs(rhop_H11)>1e-6);deg_rho_H11 = deg_rho_H11(inonzeros);rhop_H11 = rhop_H11(inonzeros);rho_H11 = rho_H11(inonzeros);

inonzeros = find(abs(lambdap_H2)>1e-6);deg_lambda_H2 = deg_lambda_H2(inonzeros);lambdap_H2 = lambdap_H2(inonzeros);
inonzeros = find(abs(rhop_H2)>1e-6);deg_rho_H2 = deg_rho_H2(inonzeros);rhop_H2 = rhop_H2(inonzeros);rho_H2 = rho_H2(inonzeros);

inonzeros = find(abs(lambda_H)>1e-6);deg_lambda_H = deg_lambda_H(inonzeros);lambda_H = lambda_H(inonzeros);lambdap_H = lambdap_H(inonzeros);
inonzeros = find(abs(rho_H)>1e-6);deg_rho_H = deg_rho_H(inonzeros);rho_H = rho_H(inonzeros);rhop_H = rhop_H(inonzeros);

gap_dB = 20*log10(thresh_th/sigma)

toc













