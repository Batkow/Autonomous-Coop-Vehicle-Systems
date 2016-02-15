clear all, close all, clc
% Load data
flist = dir('*.mat');
for i = 1:4
  tmp = load(flist(i).name,'-ascii');
  rostockData(:,i) = tmp(:,1);
end

for i = 5:8
  tmp = load(flist(i).name,'-ascii');
  ruralData(:,i-4) = tmp(:,1);
end
flist.name
clear tmp flist i
%%
clc
figure(1)
plotHandle=plot(rostockData,'linewidth',2);
h = legend('320x240','640x480','800x600','Original');
set(h,'fontsize',14,'fontweight','bold');
set(gca,'fontsize',16,'fontweight','bold');
xHandle = xlabel('Frames','fontsize',16,'fontweight','bold');
yHandle = ylabel('Offset [m]');
titleHandle = title('Autobahn video','fontsize',16,'fontweight','bold');
figure(2)
plot(ruralData,'linewidth',2);
h = legend('320x240','640x480','800x600','Original');
set(h,'fontsize',14,'fontweight','bold');
set(gca,'fontsize',16,'fontweight','bold');
xHandle = xlabel('Frames','fontsize',16,'fontweight','bold');
yHandle = ylabel('Offset [m]');
titleHandle = title('AstaZero video','fontsize',16,'fontweight','bold');

