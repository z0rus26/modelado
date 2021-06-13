clear all;
clc;
YU = dlmread('data.txt'); % Lectura del archivo
in = [YU(1:1000, 2)]; % Separacion de la entrada "u"
out = [YU(1:1000, 3)]; % Separacion de la respuesta "y"
time =  [YU(1:1000, 1)]; % Separacion de la respuesta "t"
Tc = length(out); %Tiempo de convergencia.
ts=0.08;

in = detrend(in);
out = detrend(out);
n = 10; % in and out are the same size.
[test,train]=kfolds(n,in,out);
erse=zeros(1,n);
teta=zeros(n,6);
%entrenamiento de los 10 modelos

for aa=1:n  
        x_test=test{aa,1};
        y_test=test{aa,2};
        x_train=train{aa,1};
        y_train=train{aa,2};    
        Y=[0;0;0;y_train(4:end)];
        H=[[0;0;0;y_train(3:end-1)] [0;0;0;y_train(2:end-2)] [0;0;0;y_train(1:end-3)] [0;0;0;x_train(3:end-1)] [0;0;0;x_train(2:end-2)] [0;0;0;x_train(1:end-3)]];
        tetas=((H'*H)^-1)*H'*Y;        
        tetax=tetas';
        teta(aa,:)=tetax;
        %coeficientes de tercer orden para el modelo entrado 
        sys3=tf([teta(aa,4) teta(aa,5) teta(aa,6)],[1 -teta(aa,1) -teta(aa,2) -teta(aa,3)],ts);
        %calculo del error        
        al=length(x_test);
        ttest=linspace(0,(al*ts)-ts,al);
        yd3=lsim(sys3,x_test,ttest); %salida del modelo
        erse(1,aa)=sqrt(mean((y_test-yd3).^2)); 
end
%calculo para saber cual es el mejor modelo
erse
promedio=mean(erse)
abs(erse-promedio)
mejor=min(abs(mean(erse)-erse))

%graficar el modelo #6
modelo=6;
x_test=in;
y_test=out;
teta(modelo,:)
sys3=tf([teta(modelo,4) teta(modelo,5) teta(modelo,6)],[1 -teta(modelo,1) -teta(modelo,2) -teta(modelo,3)],ts)
al=length(y_test);
ttest=linspace(0,(al*ts)-ts,al);
yd3=lsim(sys3,x_test,ttest); %salida del modelo

erse_final=sqrt(mean((y_test-yd3).^2));

figure
plot(ttest,yd3,'r','DisplayName','Modelo entrenado')
hold on
plot(ttest,y_test,'b','DisplayName','Data prueba')
xlabel('tiempo [s]')    
ylabel('Salida-temperatura [°C]')
titlee = sprintf('Comparación del mejor modelo obtenido de tercer ordeno  con Erse de %.4f % ',erse);
title(titlee)
legend('Modelo entrenado','Datos de prueba')
