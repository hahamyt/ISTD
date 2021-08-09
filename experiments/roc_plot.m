function roc_plot(fpr, tpr, precision, recall, f1score, accuracy, detectors, seqs, len_t)
    num_seq = length(seqs);
    num_det = length(detectors);
    
    a = {'r','g','b','c','m','y','k','b--','r--','c--','m--','y--','k--','g:d','b--<','c--*','m:x',...
        'g--+','k:*','y-->','r--s','g:--o','b--p','c:d','m--*','r:d','k--^','r:.'};
    %% FPR and TPR plotting    
    for s = 1:num_seq
       figure(s+1);clf;
       le = {};
       aucs = zeros(num_det, 1);
       for d = 1:num_det
          x = squeeze(fpr(d, s, :)); 
          y = squeeze(tpr(d, s, :));
          auc =0;
          for i = 2:len_t-1
            auc=auc+(y(i)+y(i-1))*(x(i-1)-x(i))/2;
          end
          auc = auc+x(len_t-1)*y(len_t-1)/2;
          auc = roundn(auc,-4);
          aucs(d)=auc;
       end
       [aucs, I] = sort(aucs, 'descend');
       for d = 1:num_det
          title(['ROC Plots on Seq' num2str(s)],'fontname','Times New Roman','Color','black','FontSize',16);
          le = [le,['[' num2str(aucs(d)) '] ' detectors{I(d)}]];
          x = squeeze(fpr(I(d), s, :)); 
          y = squeeze(tpr(I(d), s, :));
          plot(x, y, a{I(d)},'linewidth',2);
          xlabel('FPR','fontname','Times New Roman','Color','black','FontSize',14);
          ylabel('TPR','fontname','Times New Roman','Color','black','FontSize',14);
          hold on;
          grid on;
          set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',1)
       end
       plot(0:0.1:1, 0:0.1:1, 'k--','linewidth',0.5);
       legend(le,'fontname','Times New Roman','Color','w','FontSize',14);
    end
    
    %% Precision and Recall plotting
    for s = 1:num_seq
       figure(num_seq+s+1);clf;
       le = {};
       aucs = zeros(num_det, 1);
       for d = 1:num_det
          x = squeeze(recall(d, s, :));
          y = squeeze(precision(d, s, :)); 
          auc =0;
          for i = 2:len_t-1
            auc=auc+(y(i)+y(i-1))*(x(i-1)-x(i))/2;
          end
          auc = auc+x(len_t-1)*y(len_t-1)/2;
          auc = roundn(auc,-4);
          aucs(d, :) = auc;
       end
       [aucs, I] = sort(aucs, 'descend');
       for d = 1:num_det
          title(['Precison-Recall Plots on Seq' num2str(s)],'fontname','Times New Roman','Color','black','FontSize',16);
          le = [le,['[' num2str(aucs(d)) '] ' detectors{I(d)}]];
          x = squeeze(recall(I(d), s, :)); 
          y = squeeze(precision(I(d), s, :));
          plot(x, y, a{I(d)},'linewidth',2);
          xlabel('Recall','fontname','Times New Roman','Color','black','FontSize',14);
          ylabel('Precision','fontname','Times New Roman','Color','black','FontSize',14);
          hold on;
          grid on;
          set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',1)
       end
       legend(le,'fontname','Times New Roman','Color','w','FontSize',14);
    end
        
    %% F1 score
%     a = {'d','<','*','x','+','>','s','o','p','^','h', '^','v'};
%     f1score=mean(f1score, 3);
%     for s = 1:num_seq
%        figure(2*num_seq+s+1);clf;
%        le = {};
%        [f1score, I] = sort(f1score, 'descend');
%        for d = 1:num_det
%           title(['F1-score Plots on Seq' num2str(s)],'fontname','Times New Roman','Color','black','FontSize',16);
%           y = roundn(squeeze(f1score(d, s)), -4);
%           
%           le = [le,['[' num2str(y) '] ' detectors{I(d)}]];
%           scatter(num_det-d+1, y, 140, a{d},'linewidth',3);
%           xlabel('Detectors','fontname','Times New Roman','Color','black','FontSize',14);
%           ylabel('F1-score','fontname','Times New Roman','Color','black','FontSize',14);
%           hold on;
%           grid on;
%           set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',1)
%        end
%        plot(1:num_det, sort(f1score(:, s), 'ascend'), 'k--');
%        legend(le,'fontname','Times New Roman','Color','w','FontSize',14);
%     end
%     
%     %% Accuracy
%     a = {'d','<','*','x','+','>','s','o','p','^','h', '^','v'};
%     accuracy=mean(accuracy, 3);
%     for s = 1:num_seq
%        figure(3*num_seq+s+1);clf;
%        le = {};
%        [accuracy, I] = sort(accuracy, 'descend');
%        for d = 1:num_det
%           title(['Accuracy Plots on Seq' num2str(s)],'fontname','Times New Roman','Color','black','FontSize',16);
%           y = roundn(squeeze(accuracy(d, s)), -4);
%           
%           le = [le,['[' num2str(y) '] ' detectors{I(d)}]];
%           scatter(num_det-d+1, y, 140, a{d},'linewidth',3);
%           xlabel('Detectors','fontname','Times New Roman','Color','black','FontSize',14);
%           ylabel('Accuracy','fontname','Times New Roman','Color','black','FontSize',14);
%           hold on;
%           grid on;
%           set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',1)
%        end
%        plot(1:num_det, sort(accuracy(:, s), 'ascend'), 'k--');
%        legend(le,'fontname','Times New Roman','Color','w','FontSize',14);
%     end
%     
end