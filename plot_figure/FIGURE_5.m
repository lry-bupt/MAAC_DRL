X_1 = [0.02, 0.04, 0.06, 0.08, 0.1];
Y1_NOMA = [-1.333717993, -7.825136502, -10.44519537, -13.65757053, -16];
Y1_OMA = [-1.42E-05,-0.025458926,-0.356210216,-0.629268766,-0.818900052];
Y2_NOMA = [-0.271909959, -2.990650853,-5.692890644,-6.519637193,-9.841153601]; %FIX
Y2_OMA = [-2.88E-03,-0.027303647,-0.147327698,-0.395702838,-0.600764221];
Y3_NOMA = [-0.002032929,-0.391495296,-1.822073156,-2.974095926,-7.442980097];%3 U 
Y3_OMA = [-2.98E-11,-1.93E-07,-8.47E-07,-1.86E-05,-8.91E-05];
Y4_NOMA = [-0.004375222,-1.439502907,-3.918489578,-7.066198785,-9.329658858];%0.01 
Y4_OMA = [-2.25E-06,-0.000636687,-0.071654114,-0.217504705,-0.407177846];
Y5_NOMA = [-0.004367413,-2.221372461,-4.209532276,-6.103669332,-7.890873251];%ddpg 
Y5_OMA = [-4.12E-11,-9.69E-05,-0.004444109,-0.037339037,-0.142955241];
Y6_NOMA = [-0.859490241,-2.221372461,-5.558103089,-7.355402684,-8.419060732];%2 envir
Y6_OMA = [-2.57E-01,-0.906342075,-1.958136833,-2.708482246,-4.548873318];

% fig = figure;
% left_color = [0 0 0];
% right_color = [0 0 0];
% set(fig,'defaultAxesColorOrder',[left_color; right_color]);

%激活左侧
% yyaxis left
p1=plot(X_1,Y1_NOMA,'LineWidth',2, 'MarkerSize',6, 'MarkerEdgeColor', 'k', 'MarkerFaceColor','b');
hold on
p2=plot(X_1,Y1_OMA,'-o','LineWidth',2, 'MarkerSize',6, 'MarkerEdgeColor', 'k', 'MarkerFaceColor','b');
hold on
% p3=plot(X_1,Y2_NOMA,'-^','LineWidth',2, 'MarkerSize',6, 'MarkerEdgeColor', 'k', 'MarkerFaceColor','b');
% ylabel('Averaged data rate recieved by each robot')
% hold on
% % yyaxis right
% p4=plot(X_1,Y2_OMA,'LineWidth',2, 'MarkerSize',6, 'MarkerEdgeColor', 'r', 'MarkerFaceColor','b');
% hold on
% p5=plot(X_1,Y3_NOMA,'-o','LineWidth',2, 'MarkerSize',6, 'MarkerEdgeColor', 'r', 'MarkerFaceColor','b');
% hold on
% p6=plot(X_1,Y3_OMA,'-^','LineWidth',2, 'MarkerSize',6, 'MarkerEdgeColor', 'r', 'MarkerFaceColor','b');
% hold on
p1=plot(X_1,Y4_NOMA,'LineWidth',2, 'MarkerSize',6, 'MarkerEdgeColor', 'k', 'MarkerFaceColor','b');
hold on
p2=plot(X_1,Y4_OMA,'-o','LineWidth',2, 'MarkerSize',6, 'MarkerEdgeColor', 'k', 'MarkerFaceColor','b');
hold on
p3=plot(X_1,Y5_NOMA,'-^','LineWidth',2, 'MarkerSize',6, 'MarkerEdgeColor', 'k', 'MarkerFaceColor','b');
ylabel('Averaged data rate recieved by each robot')
hold on
% yyaxis right
p4=plot(X_1,Y5_OMA,'LineWidth',2, 'MarkerSize',6, 'MarkerEdgeColor', 'r', 'MarkerFaceColor','b');
hold on
p5=plot(X_1,Y6_NOMA,'-o','LineWidth',2, 'MarkerSize',6, 'MarkerEdgeColor', 'r', 'MarkerFaceColor','b');
hold on
p6=plot(X_1,Y6_OMA,'-^','LineWidth',2, 'MarkerSize',6, 'MarkerEdgeColor', 'r', 'MarkerFaceColor','b');
hold on
ylabel('Decoding Error Probability $\lg (\mathcal{P}_1)$')
xlabel('$P_{\max}$')
legend('UE distribution $1$, $\kappa_1=0.1$, MA-TD3, NOMA','UE distribution $1$, $\kappa_1=0.1$, MA-TD3, OMA','$UE distribution $1$, \kappa_1=0.01$, MA-TD3, NOMA','$UE distribution $1$, \kappa_1=0.01$, MA-TD3, OMA','$UE distribution $1$, \kappa_1=0.1$, MA-DDPG,NOMA','$UE distribution $1$, \kappa_1=0.1$, MA-DDPG,OMA','UE distribution $2$, $\kappa_1=0.1$, MA-TD3, NOMA','UE distribution $2$, $\kappa_1=0.1$, MA-TD3, OMA')
% set(get(a(1),'Ylabel'),'String','Averaged data rate of each robot')
% set(get(a(2),'Ylabel'),'String','Averaged arriving step of each robot')

box on
grid off
