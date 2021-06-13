%%
%minimos cuadrados fuera de linea
clear all;
clc;
YU = dlmread('data.txt'); % Lectura del archivo
in = [YU(1:1000, 2)]; % Separacion de la entrada "u"
out = [YU(1:1000, 3)]; % Separacion de la respuesta "y"
time =  [YU(1:1000, 1)]; % Separacion de la respuesta "t"
Tc = length(out); %Tiempo de convergencia.
ts=0.08;

%%
%%separación de de conjuntos de datos
dtrain=10;    %porcentaje de datos para entrenar modelo

tt=dtrain*Tc/100
%selección de datos para entrenar modelo
u = [YU(1:tt, 2)]; % Separacion de la entrada "u"
y = [YU(1:tt, 3)]; % Separacion de la respuesta "y"
u = detrend(u);
y = detrend(y);

T = length(y); %Tiempo de convergencia.

%selección de datos para probar modelo
utest = [YU(T+1:end, 2)]; % Separacion de la entrada "u"
ytest = [YU(T+1:end, 3)]; % Separacion de la respuesta "y"
utest = detrend(utest);
ytest = detrend(ytest);

%ttest =  [YU(dtest:end, 1)]; % Separacion de la respuesta "t"



%% primer orden 
Y=[0;y(2:end)];
H=[[0;y(1:end-1)] [0;u(1:end-1)] ];
teta=inv(H'*H)*H'*Y;
teta'

sys1=tf(teta(2),[1 -teta(1)],ts);
%yd1 es con el modelo entrenado a ser testeado con la data de test
a=length(ytest);
ttest=linspace(0,(a*ts)-ts,a);

%testeando el modelo
yd1=lsim(sys1,utest,ttest);

figure
plot(ttest,yd1,'r','DisplayName','Modelo entrenado')
hold on
plot(ttest,ytest,'b','DisplayName','Data prueba')
xlabel('tiempo [s]')
ylabel('Salida-temperatura [°C]')
titlee = sprintf('Comparación del modelo de primer orden entrenado al %.0f %% vs datos de prueba ',dtrain);
title(titlee)

legend('Modelo','Datos de prueba')
%calculo error
error1=(ytest-yd1)'*(ytest-yd1)
error1a=mean(ytest-yd1)

erse=sqrt(mean((ytest-yd1).^2))



%% segundo orden 
Y=[0;0;y(3:end)];

H=[[0;0;y(2:end-1)] [0;0;y(1:end-2)] [0;0;u(2:end-1)] [0;0;u(1:end-2)]];
teta=((H'*H)^-1)*H'*Y;
teta'

sys2=tf([teta(3) teta(4)],[1 -teta(1) -teta(2)],ts);
a=length(ytest);

ttest=linspace(0,(a*ts)-ts,a);
yd2=lsim(sys2,utest,ttest);

figure
plot(ttest,yd2,'r','DisplayName','Modelo entrenado')
hold on
plot(ttest,ytest,'b','DisplayName','Data prueba')
xlabel('tiempo [s]')
ylabel('Salida-temperatura [°C]')
titlee = sprintf('Comparación del modelo de segundo orden entrenado al %.0f %% vs datos de prueba ',dtrain);
title(titlee)

legend('Modelo','Datos de prueba')
%calculo error
error2=(ytest-yd2)'*(ytest-yd2)
error2a=mean(ytest-yd2)
erse=sqrt(mean((ytest-yd2).^2))
%% tercer orden 
Y=[0;0;0;y(4:end)];
H=[[0;0;0;y(3:end-1)] [0;0;0;y(2:end-2)] [0;0;0;y(1:end-3)] [0;0;0;u(3:end-1)] [0;0;0;u(2:end-2)] [0;0;0;u(1:end-3)]];
teta=((H'*H)^-1)*H'*Y;
teta'

sys3=tf([teta(4) teta(5) teta(6)],[1 -teta(1) -teta(2) -teta(3)],ts);
a=length(ytest);

ttest=linspace(0,(a*ts)-ts,a);
yd3=lsim(sys3,utest,ttest);
figure
plot(ttest,yd3,'r','DisplayName','Modelo entrenado')
hold on
plot(ttest,ytest,'b','DisplayName','Data prueba')
xlabel('tiempo [s]')
ylabel('Salida-temperatura [°C]')
titlee = sprintf('Comparación del modelo de tercer orden entrenado al %.0f %% vs datos de prueba ',dtrain);
title(titlee)

legend('Modelo entrenado','Datos de prueba')
%calculo error
error3=(ytest-yd3)'*(ytest-yd3)
error3a=(ytest-yd3)
erse=sqrt(mean((ytest-yd3).^2))

%% cuarto orden 
Y=y(5:end);
H=[y(4:end-1) y(3:end-2) y(2:end-3) y(1:end-4) u(4:end-1) u(3:end-2) u(2:end-3) u(1:end-4)];
teta=((H'*H)^-1)*H'*Y;
teta'

sys4=tf([teta(5) teta(6) teta(7) teta(8)],[1 -teta(1) -teta(2) -teta(3) -teta(4)],ts);
a=length(ytest);

ttest=linspace(0,(a*ts)-ts,a);
yd4=lsim(sys4,utest,ttest);
figure
plot(ttest,yd4,'r','DisplayName','Modelo entrenado')
hold on
plot(ttest,ytest,'b','DisplayName','Data prueba')
xlabel('tiempo [s]')
ylabel('Salida-temperatura [°C]')
titlee = sprintf('Comparación del modelo de cuarto orden entrenado al %.0f %% vs datos de prueba ',dtrain);
title(titlee)

legend('Modelo','Datos de prueba')
%calculo error
error4=(ytest-yd4)'*(ytest-yd4)
error4a=mean(ytest-yd4)
erse=sqrt(mean((ytest-yd4).^2))



%Validacion cruzada
%%
%minimos cuadrados fuera de linea
clear all;
clc;
YU = dlmread('data.txt'); % Lectura del archivo
in = [YU(1:1000, 2)]; % Separacion de la entrada "u"
out = [YU(1:1000, 3)]; % Separacion de la respuesta "y"
time =  [YU(1:1000, 1)]; % Separacion de la respuesta "t"
Tc = length(out); %Tiempo de convergencia.
ts=0.08;

%%
%%separación de  subconjuntos de datos
in = detrend(in);
out = detrend(out);
n=20;
ymatrix=reshape(out,Tc/n,n);
umatrix=reshape(in,Tc/n,n);
error_vector=zeros(n,n);

for aa=1:n    
    %separa los subsets de datos
    subset_y=ymatrix(1:end,aa); % subset de salidas
    subset_u=umatrix(1:end,aa); % subset de entradas
    Y=[0;0;0;subset_y(4:end)];
    H=[[0;0;0;subset_y(3:end-1)] [0;0;0;subset_y(2:end-2)] [0;0;0;subset_y(1:end-3)] [0;0;0;subset_u(3:end-1)] [0;0;0;subset_u(2:end-2)] [0;0;0;subset_u(1:end-3)]];
    teta=((H'*H)^-1)*H'*Y;
    teta';
    %coeficientes de tercer orden para el modelo entrado 
    sys3=tf([teta(4) teta(5) teta(6)],[1 -teta(1) -teta(2) -teta(3)],ts);
    
    %validación del modelo
    for dd=1:n
        
    subset_entrada_test=umatrix(1:end,dd); % subset de entradas
    %selecciona el modelo a testear las entradas
    a=length(subset_entrada_test);
    ttest=linspace(0,(a*ts)-ts,a);
    yd3=lsim(sys3,subset_entrada_test,ttest); %salida del modelo
    erse=sqrt(mean((subset_y-yd3).^2));
    error_vector(aa,dd)=erse;
    end
end




