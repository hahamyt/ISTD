function huberandf()

    x = [-6:0.1:6];
    kappa1 = 0.5;
    kappa2 = 1;
    kappa3 = 2;
    kappa4 = 3;
    
    huber_y1 = @(x) 0.5*x.^2.*(abs(x)<=kappa1) + kappa1*(abs(x)-0.5*kappa1).*(abs(x)>kappa1);
    huber_y2 = @(x) 0.5*x.^2.*(abs(x)<=kappa2) + kappa2*(abs(x)-0.5*kappa2).*(abs(x)>kappa2);
    huber_y3 = @(x) 0.5*x.^2.*(abs(x)<=kappa3) + kappa3*(abs(x)-0.5*kappa3).*(abs(x)>kappa3);
    huber_y4 = @(x) 0.5*x.^2.*(abs(x)<=kappa4) + kappa4*(abs(x)-0.5*kappa4).*(abs(x)>kappa4);
    
    mse_y = @(x) 0.5*x.^2;
    title 'Huber and F-norm estimation comparison';
    plot(x, huber_y1(x));hold on;
    plot(x, huber_y2(x));hold on;
    plot(x, huber_y3(x));hold on;
    plot(x, huber_y4(x));hold on;
    plot(x, mse_y(x));
    legend("Huber \kappa=0.5", "Huber \kappa=1", "Huber \kappa=2", "Huber \kappa=3","F-Norm")
end

