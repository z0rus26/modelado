function [test, train] = separacion_data(n,X,Y)
  %longitud de los datos
  num = size(X,1); 
  %vector de zeros para el test y prueba
  test{n,2} = [];
  train{n,2} = [];
  %redondeo de la cifra del subset  
  subset = floor(num/n);
  
  %creación del primer subset
  test{1,1} = X(1:subset,:);
  test{1,2} = Y(1:subset,:);
  train{1,1} = X(subset+1:end,:); 
  train{1,2} = Y(subset+1:end,:);  
  
  %creación de los subset restantes
  for f = 2:n
      test{f,1} = X((f-1)*subset+1:(f)*subset,:);
      train{f,1} = [X(1:(f-1)*subset,:); X(f*subset+1:end, :)];      
      test{f,2} = Y((f-1)*subset+1:(f)*subset,:);
      train{f,2} = [Y(1:(f-1)*subset,:); Y(f*subset+1:end, :)];
  end
end
