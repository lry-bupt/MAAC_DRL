clc;
clear all;
close all;
X=[1:1:100];
Y=[1:1:100];%square room:40*60m
MARK=zeros(length(X),length(Y));
MARK_PL=zeros(length(X),length(Y));
for i=1:1:length(X)
for j=1:1:length(Y)
    i2=(i-1)/2;
    j2=(j-1)/2;
% MARK(i,j)=0;
    if  (j2-20-3*i2<=0) &&(j2-i2-20>=0) && (i2>=0) &&  (j2>30) && (i2>5)%(48,60)
         MARK(i,j)=1;
    end
    if  (j2+3/4*i2-20>=0) &&(j2-20+(2/5*i2)<=0) && (j2<=20) &&  (j2>=0) && (i2>20) && (j2<10)%(72,60)
         MARK(i,j)=1;
    end
    if  (j2>=0) &&(j2<=40) && (i2>=30) &&  (i2<=60)%(72,60)
         MARK(i,j)=1;
    end
%MARK:假设3标注有遮挡物�?表示LoS�?表示NLoS,4表示AP�?为默认�?
end
end
%先标注有遮挡�?
% MARK(10,20)=1;
MARK=BLOCK(MARK,15,65,5);
MARK=BLOCK(MARK,45,15,5);
%标注AP
MARK(1,41)=4;




surf(X,Y,MARK);
axis([0 120 0 120 0 9]);%限定显示的范�?
xlabel('x�?);%x轴坐�?
ylabel('y�?);%y轴坐�?
zlabel('z�?);%z轴坐�?
title('MARK');%标题

%MARK:假设1标注有遮挡物�?表示LoS�?表示NLoS,4表示AP�?为默认�?
for i=1:1:length(X)
for j=1:1:length(Y)
    if MARK(i,j)==2
        %有遮挡物
        MARK_PL(i,j)=90;
    end
    if MARK(i,j)==4
        %有遮挡物
        MARK_PL(i,j)=40;
    end
    if MARK(i,j)==0
        MARK_PL(i,j)=(23.1*log(sqrt((i)*(i)/4+(j-20)*(j-20)/4))/log(10)+24.52+20.6*log(sqrt(2))/log(10));
        %LoS
    end
    if MARK(i,j)==1
        MARK_PL(i,j)=(37.9*log(sqrt((i)*(i)/4+(j-20)*(j-20)/4))/log(10)+21.01+13.4*log(sqrt(2))/log(10));
        %NLoS
    end
    %MARK_PL(i,j)=- MARK_PL(i,j)/10;
    MARK_PL(i,j)=power(10,(- MARK_PL(i,j)/10));
end
end
%print(MARK_PL)
figure (2)
surf(X,Y,MARK_PL);
axis([0 80 0 120]);%限定显示的范�?
xlabel('x�?);%x轴坐�?
ylabel('y�?);%y轴坐�?
zlabel('z�?);%z轴坐�?
title('MARK_P_L');%标题

function [NEW_MARK] = BLOCK(inputArg1,inputArg2,inputArg3,inputArg4)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
  for i=inputArg2-inputArg4+2:1:inputArg2+inputArg4+1
  for j=inputArg3-inputArg4+2:1:inputArg3+inputArg4+1
%     print(i)  
    inputArg1(i,j) = 2;
  end
  end
  NEW_MARK = inputArg1;
%   inputArg1(inputArg1+1,inputArg1+1) = 1;
end

