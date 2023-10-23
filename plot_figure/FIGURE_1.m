clc;
clear all;
close all;
X=[1:1:400];
TD3_1=importdata('C:\Users\Administrator.DESKTOP-NLH290A\Desktop\2302_code\FIGURE_1\MATLAB\TD3-1.txt');
TD3_2=importdata('C:\Users\Administrator.DESKTOP-NLH290A\Desktop\2302_code\FIGURE_1\MATLAB\TD3-2.txt');
TD3_3=importdata('C:\Users\Administrator.DESKTOP-NLH290A\Desktop\2302_code\FIGURE_1\MATLAB\TD3-2.txt');
TD3_avg=(TD3_1+TD3_2+TD3_3)/3;

% importdata('C:\Users\Administrator.DESKTOP-NLH290A\Desktop\2302_code\FIGURE_1\MATLAB\TD3-avg.txt');
PPO_avg=importdata('C:\Users\Administrator.DESKTOP-NLH290A\Desktop\2302_code\FIGURE_1\MATLAB\PPO-avg.txt');
DDPG_avg=importdata('C:\Users\Administrator.DESKTOP-NLH290A\Desktop\2302_code\FIGURE_1\MATLAB\DDPG-avg.txt');
SAC_avg=importdata('C:\Users\Administrator.DESKTOP-NLH290A\Desktop\2302_code\FIGURE_1\MATLAB\SAC-avg.txt');
RANDOM_avg=ones(400)*TD3_avg(1);
PPO_avg(1)=TD3_avg(1);
DDPG_avg(20)=TD3_avg(1);
SAC_avg(20)=TD3_avg(1);

p1=plot(X, TD3_avg(20:419), '-p', 'MarkerSize',5, 'LineWidth',1.2,'Color',[0.00,0.45,0.74],'MarkerIndices',1:100:400);
hold on
p2=plot(X, PPO_avg(1:400), '-o', 'MarkerSize',5, 'LineWidth',1.2,'Color',[0.47,0.67,0.19],'MarkerIndices',1:100:400);
hold on
p3=plot(X, DDPG_avg(20:419), '-.*', 'MarkerSize',5, 'LineWidth',1.5,'Color',[1	0.54902	0],'MarkerIndices',1:100:400);
hold on
p4=plot(X, SAC_avg(20:419), '-', 'MarkerSize',5, 'LineWidth',1.2,'Color',[0.50,0.50,0.50],'MarkerIndices',1:100:400);
hold on
p5=plot(X, RANDOM_avg(1:400), '--', 'MarkerSize',5, 'LineWidth',1.2,'Color',[0.72,0.27,1.00],'MarkerIndices',1:100:400);
% p1.MarkerIndices = 400:500:length(y1_ping);
legend([p1 p4 p3 p2  p5 ],{'TD3','SAC','DDPG','PPO','Random'},'Location','SouthEast','Interpreter','latex')
xlabel('Episode','Interpreter','latex')
ylabel('Reward','Interpreter','latex')
ylim([-1500, 0])