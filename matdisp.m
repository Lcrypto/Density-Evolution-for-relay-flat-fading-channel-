function matdisp(R,R_bc,t,lambdap_H11,deg_lambda_H11,lambda_H,deg_lambda_H1,deg_lambda_H2);

% Initialization

lambdap_H22 = lambda_H(length(deg_lambda_H1)+1:length(deg_lambda_H1)+length(deg_lambda_H2))./(deg_lambda_H2');lambdap_H22 = lambdap_H22/sum(lambdap_H22);

lambdap_H11 = lambdap_H11/sum(lambdap_H11);

cs_lambdap_H11 = cumsum(lambdap_H11);
lambdap_H1 = (lambda_H(1:length(deg_lambda_H1))./deg_lambda_H1');lambdap_H1 = lambdap_H1/sum(lambdap_H1);
cs_lambdap_H1 = cumsum(lambdap_H1);
cs_lambdap_H21 = sort(unique([cs_lambdap_H1;cs_lambdap_H11]));
lambdap_H21 = [cs_lambdap_H21(1);cs_lambdap_H21(2:end)-cs_lambdap_H21(1:end-1)];
deg_lambda_H21 = zeros(size(lambdap_H21));for n = 1:length(deg_lambda_H21);deg_lambda_H21(n) = deg_lambda_H1(min(find(cs_lambdap_H1>=cs_lambdap_H21(n)-(1e-6)))) - deg_lambda_H11(min(find(cs_lambdap_H11>=cs_lambdap_H21(n)-(1e-6))));end;
deg_lambda_H2 = [reshape(deg_lambda_H21,1,length(deg_lambda_H21)),reshape(deg_lambda_H2,1,length(deg_lambda_H2))];
lambdap_H2 = [t*reshape(lambdap_H21,1,length(lambdap_H21)),(1-t)*reshape(lambdap_H22,1,length(lambdap_H22))];
deg_lambda_max = max([max(deg_lambda_H1),max(deg_lambda_H2)]);

% display

N = 500;M = round((1-R)*N);Ma = round(t*(1-R_bc)*N);mat_disp = zeros(M,N);
vNa = [1;round(cumsum(N*t*lambdap_H11)/sum(lambdap_H11))];
vNb = [1,round(cumsum(N*lambdap_H2)/sum(lambdap_H2))];
vM = [round((1-R_bc)*t*N);M];
for n = 1:length(vNa)-1;
	mat_disp(1:vM(1),vNa(n):vNa(n+1)) = deg_lambda_H11(n);
end;
for n = 1:length(vNb)-1;
	mat_disp(vM(1)+1:vM(2),vNb(n):vNb(n+1)) = deg_lambda_H2(n);
end;
%mat_disp = mat_disp/max(max(mat_disp));
figure;imagesc(linspace(0,1,N),linspace(0,1-R,M),mat_disp);
h=line([0,1],[t*(1-R_bc),t*(1-R_bc)]);set(h,'LineWidth',3,'Color',[0 0 0]);h=line([t,t],[0,t*(1-R_bc)]);set(h,'LineWidth',3,'Color',[0 0 0]);
%h=line([0,1],[0,1-R]);set(h,'LineWidth',2,'Color',[0 0 0]);
h=text(t/2,t*(1-R_bc)/2,'H_{11}');set(h,'FontSize',20);h=text(1/2,t*(1-R_bc)/2+(1-R)/2,'H_{2}');set(h,'FontSize',20);h=text((t+1)/2,t*(1-R_bc)/2,'0');set(h,'FontSize',30);
colormap('jet');colorbar;

% To compare with real matrix H
%figure;imagesc(H);colormap('gray');


