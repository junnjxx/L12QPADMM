function [x,y,iter] = vector_splitadmm(A,k,vcfg,X1,Y1)
n=size(A,1); eta1=vcfg.eta; beta1=vcfg.beta; maxiter=vcfg.max_iter;
% min 1/2*x'*A*x s.t. e'*x=k xi=0or1
% min 1/2*x'*A*y+eta*sum(y.^0.5) s.t. e'*x=k 0<=y<=1 x-y=0
%eta = norm(A,"inf"); %% todo
%eta = 0;  %%
ratio_step = vcfg.print_every ;
u_r = 0.2; l_r = 0.05;
x = X1;
e = ones(n,1);
y = Y1;
lambda = zeros(n,1);
epsi = vcfg.epsi;
iter = 0;
%fprintf('%i\t%f\t%f\n',iter,norm(x-y),obj(x,y))
saved_obj = [];
error_list = [];
Ay_new = A * y; 
ny0 = n;
count = 0;
phase = 1;
obj0 = inf ;
beta = beta1;
eta = eta1;

for iter = 1:maxiter
%update X
Ay = Ay_new;
x = y-(lambda+Ay/2)/beta;
x = x-mean(x)+k/n;
xy = x-y;
% fprintf('%i\t%f\t%f\n',iter,norm(x-y,'fro'),obj(x,y))
temp_y = y;
%update Y
y_ = x+(lambda-A*x/2)/beta;
q = eta/(2*beta);
% min 1/2*(y-y_).^2+eta*y.^0.5
y = rooth(-y_,q).^2;
% if y<=0, yopt=0
% if y>=1 yopt=0or1
% if 0<y<1 yopt=0ory
y = min(y,1);
y=(y.^2*beta/2-beta*y.*y_+eta*y.^0.5<0).*y;
Ay_new = A * y;
%fprintf('%i\t%f\t%f\n',iter,norm(x-y,'fro'),obj(x,y))
% Objective history disabled for memory efficiency.
%updata Lambda
lambda = lambda+beta*(x-y);


obj1 = obj(x,y);
% 终止准则
interval = 1;
if mod(iter,interval) ==0  %每隔interval次算一下终止准则/更新beta eta
    del_y = temp_y - y;
    % XY = X-Y;  % 这里是  X^k+1 - Y^k+1 
    d_xy = max(abs( xy(:)  ));
    dy = norm(del_y,'fro');
    dxy = norm(xy,'fro');
    r_matrix = abs(0.5 .* Ay_new - 0.5 .* Ay + beta .* del_y);
    r = max(r_matrix(:));
    error = max(r,d_xy) ;
    if mod(iter,ratio_step) == 0
    fprintf('iter:%i,r:%f,d_XY:%f\n',iter,r,d_xy)
    end
    % Error history disabled for memory efficiency.
    if error < epsi && r <= epsi
        break;
    end
    
    if false
    fprintf('iter:%i, phase: %i,nnzY: %i,beta:%4.2g,eta:%4.2g,dxy:%4.2g,dy:%4.2g\n',iter,phase,nnz(y),beta,eta,dxy,dy)  
    end
    obj_part = 1/2*x'*A*y + lambda' * (x-y)   ;
    beta_part = + beta /2 * norm(x-y,'fro')^2;
    eta_part = eta*sum(y.^0.5);
    if mod(iter,ratio_step) == 0
    fprintf('iter:%i, phase: %i, obj_part:%4.2g, beta_part:%4.2g, eta_part:%4.2g, total_obj:%4.2g\n',iter,phase,obj_part,beta_part,eta_part,obj_part+beta_part+eta_part);
    end
    if nnz(y)< k + 2
        phase = 6;
    end
    if mod(iter,1) ==0   &&   iter > 1 %% 自适应调节beta eta的条件
        switch phase
            case 1 % unstable stage
            if obj1 < 0
                if obj1 < obj0 * 0.995
                    reduce = 1;
                else
                    reduce = 0;
                end
            else
                if obj1 < obj0 * 1.005
                    reduce = 1;
                else
                    reduce = 0;
                end
            end
            if reduce == 1
                count = count+1;
                if count == 10
                    phase=2;count=0;
                end
            else
                beta = beta*2;
                obj1 = obj(x,y);
                count = 0;
            end
            case 2 % reduce beta large step
            if obj1<obj0 && dxy < 0.1*dy
                beta = beta/2;
                obj1 = obj(x,y);
            else
                beta = beta*2;
                phase = 3;
                obj1 = obj(x,y);
            end
            case 3 % reduce beta small step
                    if obj1<obj0 && dxy<0.1*dy
                        beta = beta/1.1;
                        %eta = eta * 2; %%%
                        obj1 = obj(x,y);
                    else
                        beta = beta*1.1;
                        phase = 4;
                        ny0 = nnz(y);
                        obj1 = obj(x,y);
                    end
            case 4 
                if mod(iter,ratio_step) == 0
                    ny1 = nnz(y);
                    ratio = abs(ny1 - ny0) / (n);
                    fprintf('nnz0:%d, nnz1:%d, ratio:%f\n',ny0,ny1,ratio);
                    if ratio == 0 && ny1 == 0
                        phase = 1;
                    end
                        
                    if ratio >= u_r%0.02
                        beta = beta * 1.2;
                        obj1 = obj(x,y);
                    elseif ratio < l_r
                        beta = beta / 1.2;
                        obj1 = obj(x,y);
                    else 
                        phase = 5;
                        ny0_c5 = nnz(y);
                    
                    end
                    ny0 = ny1 ;
                end
            case 5 % stable stage % 主要调节eta
                if obj1 < 0
                    if obj1 < obj0 * 0.995
                        reduce = 1;
                    else
                        reduce = 0;
                    end
                else
                    if obj1 < obj0 * 1.005
                        reduce = 1;
                    else 
                        reduce = 0;
                    end
                end
                if reduce == 1
                    if mod(iter,ratio_step)==0
                        ny1_c5 = nnz(y);                             
                        ratio  = abs( (ny0_c5-ny1_c5)/(n) );
                        fprintf('nnz0:%d, nnz1:%d, ratio:%f\n',ny1_c5,ny1_c5,ratio);
                        if ratio >= u_r%0.02
                            phase = 4;
                            beta = beta*1.2;
                            eta = eta*1.2;
                        elseif ratio < l_r
                            phase = 4;
                            beta = beta/1.2;
                            eta = eta/1.2;
                        else
                         
                            beta = beta*1.2;
                            eta = eta *  1.2 ;
                            obj1 = obj(x,y);
                        end

                       
                    ny0_c5 = ny1_c5;    
                    end
                else
                    phase = 1;
                    beta = beta*2;
                    obj1 = obj(x,y);
                end
            case 6
                
        end
    end
    obj0 = obj1;

end






end  % for iter = 1:maxiter


    function fun = obj(x,y)
        fun = 1/2*x'*A*y+eta*sum(y.^0.5) + lambda' * (x-y) + beta /2 * norm(x-y,'fro')^2;
    end


end  % function


%%
function y =  rooth(p,q)
% x^3+px+q=0 the maximal root
%fprintf('%f,%f \n',p,q)
if q==0
    y=max(0,-p).^0.5;
else
delta=(q/2).^2+(p/3).^3;
d=(delta).^0.5;
a1=(-q/2+d).^(1/3);
a2=(-q/2-d).^(1/3);
w=(3^0.5*1i-1)/2;
y1=a1+a2;
y2=w*a1+w^2*a2;
y3=w^2*a1+w*a2;
% n1=norm(imag(y1));
y=max(real(y1),max(real(y2),real(y3))).*(delta<0);
% y=real(y);
end
end