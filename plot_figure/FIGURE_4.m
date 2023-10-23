y1_0=importdata("C:/Users/Administrator.DESKTOP-NLH290A/Downloads/matdisk/TD3-STEP/STEP_v1_agent1_0.txt").';
y2_0=importdata("C:/Users/Administrator.DESKTOP-NLH290A/Downloads/matdisk/TD3-STEP/STEP_v1_agent1_1.txt").';
y3_0=importdata("C:/Users/Administrator.DESKTOP-NLH290A/Downloads/matdisk/TD3-STEP/STEP_v1_agent1_2.txt").';
y4_0=importdata("C:/Users/Administrator.DESKTOP-NLH290A/Downloads/matdisk/TD3-STEP/STEP_v1_agent1_3.txt").';
y5_0=importdata("C:/Users/Administrator.DESKTOP-NLH290A/Downloads/matdisk/TD3-STEP/STEP_v1_agent1_4.txt").';
h4=cdfplot(y1_0(200:500));
hold on
% h2=cdfplot(y2_0(200:500));
% hold on
h2=cdfplot(y3_0(200:500));
% hold on
% h4=cdfplot(y4_0(200:500));
% hold on
h3=cdfplot(y5_0(200:500));
hold on

y1_0=importdata("C:/Users/Administrator.DESKTOP-NLH290A/Downloads/matdisk/TD3-STEP-v2/STEP_v1_agent1_0.txt").';
y2_0=importdata("C:/Users/Administrator.DESKTOP-NLH290A/Downloads/matdisk/TD3-STEP-v2/STEP_v1_agent1_1.txt").';
y3_0=importdata("C:/Users/Administrator.DESKTOP-NLH290A/Downloads/matdisk/TD3-STEP-v2/STEP_v1_agent1_2.txt").';
y4_0=importdata("C:/Users/Administrator.DESKTOP-NLH290A/Downloads/matdisk/TD3-STEP-v2/STEP_v1_agent1_3.txt").';
y5_0=importdata("C:/Users/Administrator.DESKTOP-NLH290A/Downloads/matdisk/TD3-STEP-v2/STEP_v1_agent1_4.txt").';
h5=cdfplot(y1_0(200:500));
hold on
h6=cdfplot(y2_0(200:500));
hold on
% h3=cdfplot(y3_0(200:500));
% hold on
% h4=cdfplot(y4_0(200:500));
% hold on
h1=cdfplot(y5_0(200:500));
hold on
% h6=cdfplot(y4(1500:4999));
% hold on
set(h1,'Color',[0,0,0],'LineWidth',1.2, 'MarkerSize',4,'MarkerIndices',1:60:300,'LineStyle','--')
set(h2,'Color',[0,0,0],'Marker','^','LineWidth',1.2, 'MarkerSize',4,'MarkerIndices',1:60:300,'LineStyle','--')
set(h3,'Color',[0,0,0],'Marker','*','LineWidth',1.2, 'MarkerSize',6,'MarkerIndices',1:60:300,'LineStyle','--')
set(h4,'Color',[0,0,0],'LineWidth',1.2, 'MarkerSize',4,'MarkerIndices',1:60:300)
set(h5,'Color',[0,0,0],'Marker','^','LineWidth',1.2, 'MarkerSize',4,'MarkerIndices',1:60:300)
set(h6,'Color',[0,0,0],'Marker','*','LineWidth',1.2, 'MarkerSize',6,'MarkerIndices',1:60:300)

ax1 = gca;
% set(gca,'XAxisLocation','top')
% set(gca,'YAxisLocation','right')

xlim([100 350]);
ylim([0 0.85]);
xlabel('Arriving step','Interpreter','latex')
ylabel('Cumulative distribution function','Interpreter','latex')
set(gca,'XTickLabel',{'$90 \%$','$100 \%$','$110 \%$','$120 \%$','$130 \%$','$140 \%$'});
set(gca,'YTickLabel',{'0','0.2','0.4','0.6','0.8','1.0'});
set(gca,'xtick',100:250/5:350)   
set(gca,'ytick',0:0.85/5:0.85)

% 去掉上面和右面边框上的刻度 保留边框
% box off;
grid off

xl=xlim;
yl=ylim;
% line([xl(1),xl(2)],[yl(2),yl(2)],'color',[0 0 0]);   %画上边框，线条的颜色设置为黑色
% line([xl(2),xl(2)],[yl(1),yl(2)],'color',[0 0 0]);    %画右边框 ，线条的颜色设置为黑色
gs=legend([h1 h2 h3 h4 h5 h6],{'$P_{\max}=0.02, \kappa_1=0.1$','$P_{\max}=0.04, \kappa_1=0.1$','$P_{\max}=0.1,  \kappa_1=0.1$','$P_{\max}=0.02,  \kappa_1=0.01$','$P_{\max}=0.04, \kappa_1=0.01$'},'Interpreter','latex','Location','northwest','NumColumns',1);
% set(gs,'Location',best)
title(" ")

% ax2=axes('Position',get(ax1,'Position'),...
%            'XAxisLocation','top',...
%            'YAxisLocation','left',...
%            'Color','none',...
%            'XColor','b','YColor','b');
% hold on
% 
% X_1 = [0,  0.02, 0.04, 0.06, 0.08, 0.1];
% % Y1_1 = [73.32247929, 73.59625821, 73.87003713, 73.58897458, 73.54327335, 73.57012282, 73.48700371, 73.74321622, 73.56069694, 73.43016281, 73.41759497];
% % Y1_2 = [77.09425878, 77.96301057, 78.32762068, 78.77320765, 78.1179663, 78.80548415, 78.4038846, 78.67466438, 78.16966581, 78.41473865, 78.54641531];
% % Y1_3 = [82.97800628, 84.27606398, 84.41930877, 83.87746358, 83.93316195, 84.33647529, 84.48843188, 84.8706084, 84.52499286, 85.03199086, 84.51185376];
% Y2_1 = [70.94397619, 80.87746462, 83.66667014, 85.63058596, 86.90541567, 87.85830977, 88.64414142, 89.31973965, 89.89561792, 90.38694579, 90.87500257];
% Y2_2 = [70.96419081, 80.91497367, 83.69259198, 85.64478655, 86.8985237, 87.87413717, 88.68100766, 89.43362594, 89.9297314, 90.50355753, 90.8866812];
% Y2_3 = [70.99211407, 80.94282523, 83.75781121, 85.71406009, 87.01403988, 87.92457729, 88.7456004, 89.4045163, 90.00031461, 90.44857253, 90.92024851];
% p4=plot(X_1,Y2_1,'b:o','LineWidth',1.2, 'MarkerSize',4, 'MarkerEdgeColor', 'b', 'MarkerFaceColor','b');
% p5=plot(X_1,Y2_2,'b-.*','LineWidth',1.2, 'MarkerSize',6, 'MarkerEdgeColor', 'b', 'MarkerFaceColor','b');
% p6=plot(X_1,Y2_3,'b--^','LineWidth',1.2, 'MarkerSize',4, 'MarkerEdgeColor', 'b', 'MarkerFaceColor','b');
% ylabel('Average sum-rate','Interpreter','latex')
% xlabel('Power budget $P_{\max}$','Interpreter','latex')
% % legend([p4 p5 p6],{'$\kappa_1=0.0001$','$\kappa_1=0.002$','$\kappa_1=0.005$'},'Location','NorthWest','Interpreter','latex','NumColumns',1)
% % legend('$\kappa_1=0.002$','$\kappa_1=0.005$','$\kappa_1=0.0001$')
% xlim([0 1]);
% % set(gca,'XTickLabel',{'0','0.2','0.4','0.6','0.8','1'});
% set(gca,'XAxisLocation','bottom')
% set(gca,'xtick',0:0.2:1)
% grid off
