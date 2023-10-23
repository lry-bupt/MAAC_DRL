% X = [ 8  8   12   12]*6;
% Y = [8 12 12 8]*6;
% Z = [0.5 0.5 0.5 0.5];
% p6=plot(10*6,10*6,'s','MarkerEdgeColor',[0.41176 0.41176 0.41176], 'MarkerFaceColor',[0.41176 0.41176 0.41176],'MarkerSize',10, 'LineWidth',2);
% 
% patch(Y,X,Z,[0.41176 0.41176 0.41176])
% hold on
% 
% X = [ 28  28   32   32]*6;
% Y = [8 12 12 8]*6;
% Z = [0.5 0.5 0.5 0.5];
% 
% patch(Y,X,Z,[0.41176 0.41176 0.41176])
% 
% X = [ 28  28   32   32]*6;
% Y = [28 32 32 28]*6;
% Z = [0.5 0.5 0.5 0.5];
% 
% patch(Y,X,Z,[0.41176 0.41176 0.41176])
% 
% X = [ 8  8   12   12]*6;
% Y = [28 32 32 28]*6;
% Z = [0.5 0.5 0.5 0.5];
% 
% patch(Y,X,Z,[0.41176 0.41176 0.41176])
% 
% view(90,90)
% p5=plot(0*6,15*6,'p','MarkerEdgeColor',[1	1	0 ], 'MarkerFaceColor',[1	1	0 ],'MarkerSize',5, 'LineWidth',2);
% hold on
% plot(5*6,17*6,'p','MarkerEdgeColor',[1	1	0 ], 'MarkerFaceColor',[1	1	0 ],'MarkerSize',5, 'LineWidth',2)
% hold on
% plot(13*6,17*6,'p','MarkerEdgeColor',[1	1	0 ], 'MarkerFaceColor',[1	1	0 ],'MarkerSize',5, 'LineWidth',2)
% hold on
% plot(23*6,1*6,'p','MarkerEdgeColor',[1	1	0 ], 'MarkerFaceColor',[1	1	0 ],'MarkerSize',5, 'LineWidth',2)
% hold on
% plot(31*6,3*6,'p','MarkerEdgeColor',[1	1	0 ], 'MarkerFaceColor',[1	1	0 ],'MarkerSize',7, 'LineWidth',2)
% hold on
% plot(35*6,17*6,'p','MarkerEdgeColor',[1	1	0 ], 'MarkerFaceColor',[1	1	0 ],'MarkerSize',7, 'LineWidth',2)
% hold on
% %
% plot(20*6,35*6,'p','MarkerEdgeColor',[1	1	0 ], 'MarkerFaceColor',[1	1	0 ],'MarkerSize',5, 'LineWidth',2);
% hold on
% plot(25*6,37*6,'p','MarkerEdgeColor',[1	1	0 ], 'MarkerFaceColor',[1	1	0 ],'MarkerSize',5, 'LineWidth',2)
% hold on
% plot(35*6,37*6,'p','MarkerEdgeColor',[1	1	0 ], 'MarkerFaceColor',[1	1	0 ],'MarkerSize',5, 'LineWidth',2)
% hold on
% plot(5*6,23*6,'p','MarkerEdgeColor',[1	1	0 ], 'MarkerFaceColor',[1	1	0 ],'MarkerSize',5, 'LineWidth',2)
% hold on
% plot(10*6,25*6,'p','MarkerEdgeColor',[1	1	0 ], 'MarkerFaceColor',[1	1	0 ],'MarkerSize',7, 'LineWidth',2)
% hold on
% plot(15*6,25*6,'p','MarkerEdgeColor',[1	1	0 ], 'MarkerFaceColor',[1	1	0 ],'MarkerSize',7, 'LineWidth',2)
% hold on

radio_map_=-load("C:/Users/Administrator.DESKTOP-NLH290A/Desktop/2302_code/FIGURE_3/mapdata_0717.mat").MARK_PL;
% radio_map=rot90(radio_map_);
img=imagesc(radio_map_);%画图
% axis xy
%imrotated_img = imrotate(img, 90, 'bilinear');
colorbar;
hold on
% figure(2)
y1=importdata("C:/Users/Administrator.DESKTOP-NLH290A/Desktop/2302_code/FIGURE_3/路径短/y_k169_38.txt").';
x1=importdata("C:/Users/Administrator.DESKTOP-NLH290A/Desktop/2302_code/FIGURE_3/路径短/x_k169_38.txt").';
y2=importdata("C:/Users/Administrator.DESKTOP-NLH290A/Desktop/2302_code/FIGURE_3/路径短/y_k2131_35.txt").';
x2=importdata("C:/Users/Administrator.DESKTOP-NLH290A/Desktop/2302_code/FIGURE_3/路径短/x_k2131_35.txt").';
y3=importdata("C:/Users/Administrator.DESKTOP-NLH290A/Desktop/2302_code/FIGURE_3/路径短/y_k3255_44.txt").';
x3=importdata("C:/Users/Administrator.DESKTOP-NLH290A/Desktop/2302_code/FIGURE_3/路径短/x_k3255_44.txt").';


y4=importdata("C:/Users/Administrator.DESKTOP-NLH290A/Desktop/2302_code/FIGURE_3/路径短+SINR/y_k1428_95.txt").';
x4=importdata("C:/Users/Administrator.DESKTOP-NLH290A/Desktop/2302_code/FIGURE_3/路径短+SINR/x_k1428_95.txt").';
y5=importdata("C:/Users/Administrator.DESKTOP-NLH290A/Desktop/2302_code/FIGURE_3/路径短+SINR/y_k2424_32.txt").';
x5=importdata("C:/Users/Administrator.DESKTOP-NLH290A/Desktop/2302_code/FIGURE_3/路径短+SINR/x_k2424_32.txt").';
y6=importdata("C:/Users/Administrator.DESKTOP-NLH290A/Desktop/2302_code/FIGURE_3/路径短+SINR/y_k3483_48.txt").';
x6=importdata("C:/Users/Administrator.DESKTOP-NLH290A/Desktop/2302_code/FIGURE_3/路径短+SINR/x_k3483_48.txt").';
p5=plot(50,17,'p','MarkerEdgeColor',[1	0	0], 'MarkerFaceColor',[1	0	0],'MarkerSize',10,'LineWidth',2);
hold on
plot(50,50, 'p','MarkerEdgeColor',[1	0	0], 'MarkerFaceColor',[1	0	0],'MarkerSize',10,'LineWidth',2)
hold on
plot(50,84, 'p','MarkerEdgeColor',[1	0	0], 'MarkerFaceColor',[1	0	0],'MarkerSize',10,'LineWidth',2)
hold on
for i=1:1:length(x1)
   p1=plot(y1(i),x1(i),'o','MarkerSize',3,'MarkerEdgeColor',[0.93,0.69,0.13], 'MarkerFaceColor',[0.93,0.69,0.13]); 
end
for i=1:1:length(x2)
   plot(y2(i),x2(i),'o','MarkerSize',3,'MarkerEdgeColor',[0.85,0.33,0.10], 'MarkerFaceColor',[0.85,0.33,0.10]) ;
end
for i=1:1:length(x3)
   plot(y3(i),x3(i),'o','MarkerSize',4,'MarkerEdgeColor',[1.00,0.41,0.16], 'MarkerFaceColor',[1.00,0.41,0.16]) ;
end
for i=1:1:length(x4)
   p2=plot(y4(i),x4(i),'^','MarkerSize',3,'MarkerEdgeColor',[0.76,0.43,0.96], 'MarkerFaceColor',[0.76,0.43,0.96]); 
end
for i=1:1:length(x5)
   plot(y5(i),x5(i),'^','MarkerSize',4,'MarkerEdgeColor',[0.40,0.14,0.58], 'MarkerFaceColor',[0.40,0.14,0.58]) 
end
for i=1:1:length(x6)
   plot(y6(i),x6(i),'^','MarkerSize',4,'MarkerEdgeColor',[0.58,0.27,0.78], 'MarkerFaceColor',[0.58,0.27,0.78]) 
end
hold on
p3=plot(y1(1),x1(1),'+','MarkerEdgeColor',[0.50,0.50,0.50], 'MarkerFaceColor',[0,0,0],'MarkerSize',13, 'LineWidth',3);
hold on
p4=plot(y1(length(x1)),x1(length(x1)),'X','MarkerEdgeColor',[0.50,0.50,0.50], 'MarkerFaceColor',[0,0,0],'MarkerSize',13, 'LineWidth',3);
hold on
plot(y2(1),x2(1),'+','MarkerEdgeColor',[0	0	0], 'MarkerFaceColor',[0	0	0],'MarkerSize',13,'LineWidth',3);
hold on
plot(y2(length(x2)),x2(length(x2)),'X','MarkerEdgeColor',[0	0	0], 'MarkerFaceColor',[0	0	0],'MarkerSize',13,'LineWidth',3)
hold on
plot(y3(1),x3(1),'+','MarkerEdgeColor',[0.31,0.30,0.30], 'MarkerFaceColor',[0.31,0.30,0.30],'MarkerSize',13, 'LineWidth',3);
hold on
plot(y3(length(x3)),x3(length(x3)),'+','MarkerEdgeColor',[0.31,0.30,0.30], 'MarkerFaceColor',[0.31,0.30,0.30],'MarkerSize',13, 'LineWidth',3);
hold on

% rot90;
view(-90,90);
axis equal
axis([0 100 0 100]);
legend([p1 p2],{'Distance-Aware','Communication-Aware'},'Location','NorthWest','Interpreter','latex','NumColumns',2)
set(gca,'XTick',0:20:100)
set(gca,'XTickLabel',{'0','10','20','30','40','50'})
set(gca,'YTick',0:20:100)
set(gca,'YTickLabel',{'0','10','20','30','40','50'})
xlabel('$x \ ({\rm  m})$','Interpreter','latex')
ylabel('$y \ ({\rm  m})$','Interpreter','latex')
box on
% grid on
ah=axes('position',get(gca,'position'), 'visible','off');
legend(ah,[p3 p4 p5],{'MU Starting Point ','MU Destination','SU'},'Location','NorthEast','Interpreter','latex','NumColumns',4)

