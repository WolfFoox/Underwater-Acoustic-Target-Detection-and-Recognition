%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function:
%        批量提取小波能量特征并将数据存储在excel表格中
%        
% Author:
%        LiXinRui
% Version:
%        V1     2024.10.11
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
fs = 1024;  % 采样频率
f1 = 100;   % 信号的第一个频率
f2 = 300;   % 信号第二个频率
t = 0:1/fs:1;
s = sin(2*pi*f1*t) + sin(2*pi*f2*t);  % 生成混合信号


% x_input=x_train(:,1,1);           %输入数据
path="F:\document\海天探索者\水下信标\波形";
Files = dir(fullfile(path,'*.csv'));
n=length(Files);
data=cell(n,9);

for index=1:n
    clearvars -except data index Files path
    close all
    signal_name=Files(index).name;
    data(index,1)={signal_name};
    table = readtable(path+'\'+signal_name);
    try
        x_input=table.Var2;
    catch ME
        disp(ME);
        x_input=table.CH2;
    end
    % x_input=s;
    plot(x_input);title('输入信号时域图像')  %绘制输入信号时域图像
    
    %% 查看频谱范围
    x=x_input;        %   查看频谱范围
    fs=50000*2;
    
    N=length(x); %采样点个数
    signalFFT=abs(fft(x,N));%真实的幅值
    Y=2*signalFFT/N;
    f=(0:N/2)*(fs/N);
    figure;plot(f,Y(1:N/2+1));
    ylabel('amp'); xlabel('frequency');title('输入信号的频谱');grid on
    
    %%
    wpt(index)=wpdec(x_input,3,'dmey');      %进行4层小波包分解
    % plot(wpt);                        %绘制小波包树
     
    %% 实现对节点顺序按照频率递增进行重排序
    % nodes=get(wpt,'tn');  %小波包分解系数　为什么ｎｏｄｅｓ是[7;8;9;10;11;12;13;14]
    % N_cfs=length(nodes);  %小波包系数个数
    nodes=[7;8;9;10;11;12;13;14];
    ord=wpfrqord(nodes);  %小波包系数重排，ord是重排后小波包系数索引构成的矩阵　如3层分解的[1;2;4;3;7;8;6;5]
    nodes_ord=nodes(ord); %重排后的小波系数
     
    %% 实现对节点小波节点进行重构 
    for i=1:8
        rex3(:,i)=wprcoef(wpt(index),nodes_ord(i));         
    end
     
    %% 绘制第3层各个节点分别重构后信号的频谱
    figure;                         
    for i=0:7
        subplot(2,4,i+1);
        x_sign= rex3(:,i+1); 
        N=length(x_sign); %采样点个数
        signalFFT=abs(fft(x_sign,N));%真实的幅值
        Y=2*signalFFT/N;
        f=(0:N/2)*(fs/N);
        plot(f,Y(1:N/2+1));
        ylabel('amp'); xlabel('frequency');grid on
        % axis([0 50 0 0.03]); 
        title(['小波包第3层',num2str(i),'节点信号频谱']);
    end
     
    %% wavelet packet coefficients. 求取小波包分解的各个频段的小波包系数
    cfs3_0=wpcoef(wpt(index),nodes_ord(1));  %对重排序后第3层0节点的小波包系数0-8Hz
    cfs3_1=wpcoef(wpt(index),nodes_ord(2));  %对重排序后第3层0节点的小波包系数8-16Hz
    cfs3_2=wpcoef(wpt(index),nodes_ord(3));  %对重排序后第3层0节点的小波包系数16-24Hz
    cfs3_3=wpcoef(wpt(index),nodes_ord(4));  %对重排序后第3层0节点的小波包系数24-32Hz
    cfs3_4=wpcoef(wpt(index),nodes_ord(5));  %对重排序后第3层0节点的小波包系数32-40Hz
    cfs3_5=wpcoef(wpt(index),nodes_ord(6));  %对重排序后第3层0节点的小波包系数40-48Hz
    cfs3_6=wpcoef(wpt(index),nodes_ord(7));  %对重排序后第3层0节点的小波包系数48-56Hz
    cfs3_7=wpcoef(wpt(index),nodes_ord(8));  %对重排序后第3层0节点的小波包系数56-64Hz
    %% 求取小波包分解的各个频段的能量
    E_cfs3_0=norm(cfs3_0,2)^2;  %% 1-范数：就是norm(...,1)，即各元素绝对值之和；2-范数：就是norm(...,2)，即各元素平方和开根号；
    E_cfs3_1=norm(cfs3_1,2)^2;
    E_cfs3_2=norm(cfs3_2,2)^2;
    E_cfs3_3=norm(cfs3_3,2)^2;
    E_cfs3_4=norm(cfs3_4,2)^2;
    E_cfs3_5=norm(cfs3_5,2)^2;
    E_cfs3_6=norm(cfs3_6,2)^2;
    E_cfs3_7=norm(cfs3_7,2)^2;
    E_total=E_cfs3_0+E_cfs3_1+E_cfs3_2+E_cfs3_3+E_cfs3_4+E_cfs3_5+E_cfs3_6+E_cfs3_7;
     
    p_node(1)= 100*E_cfs3_0/E_total;           % 求得每个节点的占比
    p_node(2)= 100*E_cfs3_1/E_total;           % 求得每个节点的占比
    p_node(3)= 100*E_cfs3_2/E_total;           % 求得每个节点的占比
    p_node(4)= 100*E_cfs3_3/E_total;           % 求得每个节点的占比
    p_node(5)= 100*E_cfs3_4/E_total;           % 求得每个节点的占比
    p_node(6)= 100*E_cfs3_5/E_total;           % 求得每个节点的占比
    p_node(7)= 100*E_cfs3_6/E_total;           % 求得每个节点的占比
    p_node(8)= 100*E_cfs3_7/E_total;           % 求得每个节点的占比
     
    data(index,2:9)={p_node(1),p_node(2),p_node(3),p_node(4),p_node(5),p_node(6),p_node(7),p_node(8)};
    figure;
    x=1:8;
    bar(x,p_node);
    title('各个频段能量所占的比例');
    xlabel('频率 Hz');
    ylabel('能量百分比/%');
    for j=1:8
    text(x(j),p_node(j),num2str(p_node(j),'%0.2f'),...
        'HorizontalAlignment','center',...
        'VerticalAlignment','bottom')
    end
    % E = wenergy(wpt);       %求取各个节点能量
     
    %% 绘制重构前各个特征频段小波包系数的图形
    figure;
    subplot(4,1,1);
    plot(x_input);
    title('原始信号');
    subplot(4,1,2);
    plot(cfs3_0);
    title(['层数 ',num2str(3) '  节点 0的小波0-6.25KHz',' 系数'])
    subplot(4,1,3);
    plot(cfs3_5);
    title(['层数 ',num2str(3) '  节点 5的小波31.25-37.5KHz',' 系数'])
    subplot(4,1,4);
    plot(cfs3_6);
    title(['层数 ',num2str(3) '  节点 6的小波37.5-43.75KHz',' 系数'])
     
    %% 绘制重构后时域各个特征频段的图形
    figure;
    subplot(3,1,1);
    plot(rex3(:,1));
    title('重构后0-6.25KHz频段信号');
    subplot(3,1,2);
    plot(rex3(:,6));
    title('重构后31.25-37.5KHz频段信号')
    subplot(3,1,3);
    plot(rex3(:,7));
    title('重构后37.5-43.75KHz频段信号');
end

xlswrite(path+'\data.xlsx',data);