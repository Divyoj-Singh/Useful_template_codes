clear all
[x,v,s,h,f] = parameters; 
a = x(7,:); %bifurcation parameter
b = x(1,:); %x(1) = A, x(2) = B
c = a;

%% Based on eigenvalues to judge stable vs. unstable states
ind = zeros(1,4);

snum = size(f);
num = snum(2);
j = 1;
n = 1;

for n = 1:1:(num-1)
    x1 = find(f(:,n) > 0);
    x2 = find(f(:,n+1) > 0);
    if isempty(x1) && ~isempty(x2)
        ind(j) = n + 1;
        j = j + 1;
    elseif ~isempty(x1) && isempty(x2)
        ind(j) = n + 1;
        j = j + 1;
    end
end

%%
figure1 = figure('Color',[1 1 1],'units','normalized','outerposition',[0 0 1 1]);
% set(gcf, 'Position',  [100, 100, 500, 400]);
axes1 = axes;
hold(axes1,'on');
plot(c(1:ind(1)),b(1:ind(1)),'b','LineWidth',2);

plot(c(ind(1)+1:ind(2)),b(ind(1)+1:ind(2)),'r--','LineWidth',2);
plot(c(ind(2)+1:ind(3)),b(ind(2)+1: ind(3)),'b','LineWidth',2);
plot(c(ind(3)+1:ind(4)),b(ind(3)+1:ind(4)),'r--','LineWidth',2);
plot(c(ind(4)+1:end),b(ind(4)+1:end),'b','LineWidth',2);

ylim([0 100]);
ylabel('A','FontName','Arial');

% Create xlabel
xlabel('Gamma A','FontName','Arial');
box(axes1,'on');
% Set the remaining axes properties
set(axes1,'FontName','Arial','FontSize',18);

%saveas(gcf,"bifurcation_zeb.png")
