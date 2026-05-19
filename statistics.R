library(stringr)
library(lme4)
library(lmerTest)
library(ggplot2)
library(ggthemes)
#library(readr)
if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  exdir = dirname(rstudioapi::getSourceEditorContext()$path)
  setwd(exdir)
}

dirPath = file.path("data", "r_inputs")
figurePath = file.path("results", "figures")
dir.create(figurePath, recursive = TRUE, showWarnings = FALSE)
files = list.files(dirPath)
#files = 
  for (file in files) {
    name = stringr::str_replace(file, '.csv', '')
    df = read.csv(file.path(dirPath, file))
    df$timescale = unlist(lapply(df$time, function(x){stringr::str_split(x, '_')[[1]][1]}))
    df$timescale[df$timescale %in% c("V4","V5")] = "V4_V5"
    df$timescale = factor(df$timescale)
    df$stimulation=as.character(df$stimulation)
    df$stimulation[df$stimulation=="OFF"] = "off"
    df$stimulation[df$stimulation=="OMNI"] = "oDBS"
    df$stimulation[df$stimulation=="DIR"] = "dDBS"
    df$stimulation = ordered(df$stimulation, levels = c("off", "oDBS", "dDBS"))
    assign(name, df)
    
  }
SDR_long[(SDR_long$ID==0 & SDR_long$time=="V4"),grepl("TF",colnames(SDR_long))]=NA #exclude patient 0 session V4 ephys (excessive movement artefact)
VDR_long[(VDR_long$ID==0 & VDR_long$time=="V4"),grepl("TF",colnames(VDR_long))]=NA #exclude patient 0 session V4 ephys (excessive movement artefact)
{
#}
}


#####
## performance main and interaction effects
#####
#performance SDR
sdracc = ggplot(SDR_long[SDR_long$stimulation!="off",], aes(x=stimulation, y=taskAccNorm,group=stimulation)) + 
  stat_summary(position=position_dodge(width=.3), fun.data = "mean_cl_boot") + 
  theme_minimal() + theme(panel.grid = element_blank()) + ggtitle("SDR")+geom_jitter() + ylab("Accuracy\n(Difference to baseline)")

sdrrt = ggplot(SDR_long[SDR_long$stimulation!="off",], aes(x=stimulation, y=taskRTNorm,group=stimulation)) + 
  stat_summary(position=position_dodge(width=.3), fun.data = "mean_cl_boot") +
  theme_minimal() + theme(panel.grid = element_blank()) + ggtitle("SDR")+geom_jitter()+
   ylab("RT [s]\n(Difference to baseline)")

#performance VDR
vdracc = ggplot(VDR_long[VDR_long$stimulation!="off",], aes(x=stimulation, y=taskAccNorm,group=stimulation)) + 
  stat_summary(position=position_dodge(width=.3), fun.data = "mean_cl_boot") + 
  theme_minimal() + theme(panel.grid = element_blank()) + ggtitle("VDR")+geom_jitter()+ ylab("Accuracy\n(Difference to baseline)")
vdrrt = ggplot(VDR_long[VDR_long$stimulation!="off",], aes(x=stimulation, y=taskRTNorm,group=stimulation)) +
  stat_summary(position=position_dodge(width=.3), fun.data = "mean_cl_boot") + 
  theme_minimal() + theme(panel.grid = element_blank()) + ggtitle("VDR")+geom_jitter()+ ylab("RT [s]\n(Difference to baseline)")


library(patchwork)
(sdracc + sdrrt + vdracc + vdrrt)
ggsave(file.path(figurePath, 'norm_performance.png'), width=6,height = 6)
# 
# 
# #oddball SDR
# sdrodbacc = ggplot(SDR_long[SDR_long$stimulation!="OFF",], aes(x=stimulation, y=oddballAccNorm, group=stimulation)) + 
#   stat_summary(position=position_dodge(width=.3), fun.data = "mean_cl_boot") + theme_clean() + ggtitle("SDR")
# sdrodbrt = ggplot(SDR_long[SDR_long$stimulation!="OFF",], aes(x=stimulation, y=oddballRTNorm, group=stimulation)) + 
#   stat_summary(position=position_dodge(width=.3), fun.data = "mean_cl_boot") + theme_clean() + ggtitle("SDR")
# 
# 
# #oddball VDR
# vdrodbacc = ggplot(VDR_long[VDR_long$stimulation!="OFF",], aes(x=stimulation, y=oddballAccNorm, group=stimulation)) + 
#   stat_summary(position=position_dodge(width=.3), fun.data = "mean_cl_boot") + theme_clean() + ggtitle("VDR")
# vdrodbrt = ggplot(VDR_long[VDR_long$stimulation!="OFF",], aes(x=stimulation, y=oddballRTNorm, group=stimulation)) + 
#   stat_summary(position=position_dodge(width=.3), fun.data = "mean_cl_boot") + theme_clean() + ggtitle("VDR")
# 
# 
# (sdrodbacc+geom_jitter() + sdrodbrt+geom_jitter()) / (vdrodbacc+geom_jitter() + vdrodbrt+geom_jitter())

tasks = c('VDR', 'SDR')
vars = c('taskAccNorm', 'taskRTNorm', 'oddballAccNorm', 'oddballRTNorm', 'sensitivityNorm')#,



mean(SDR_long$taskRT)
mean(VDR_long$taskRT)
sd(SDR_long$taskRT)
sd(VDR_long$taskRT)

median(SDR_long$taskAcc)
median(VDR_long$taskAcc)
min(SDR_long$taskAcc)
min(VDR_long$taskAcc)
max(SDR_long$taskAcc)
max(VDR_long$taskAcc)

m1.SDRRT=lmer(taskRTNorm ~ stimulation + (1|ID), data = SDR_long[SDR_long$time!="V0",])
m0.SDRRT=lmer(taskRTNorm ~ 1 + (1|ID), data = SDR_long[SDR_long$time!="V0",])
summary(m1.SDRRT)
exp(.5*diff(anova(m0.SDRRT,m1.SDRRT)$BIC)) #Bayes Factor 01


m1.SDRACC=lmer(taskAccNorm ~ stimulation + (1|ID), data = SDR_long[SDR_long$time!="V0",])
m0.SDRACC=lmer(taskAccNorm ~ 1 + (1|ID), data = SDR_long[SDR_long$time!="V0",])

summary(m1.SDRACC)
exp(.5*diff(anova(m0.SDRACC,m1.SDRACC)$BIC)) #Bayes Factor 01

m1.VDRRT=lmer(taskRTNorm ~ stimulation + (1|ID), data = VDR_long[VDR_long$time!="V0",])
m0.VDRRT=lmer(taskRTNorm ~ 1 + (1|ID), data = VDR_long[VDR_long$time!="V0",])
summary(m1.VDRRT)
exp(.5*diff(anova(m0.VDRRT,m1.VDRRT)$BIC)) #Bayes Factor 01

m1.VDRACC=lmer(taskAccNorm ~ stimulation + (1|ID), data = VDR_long[VDR_long$time!="V0",])
m0.VDRACC=lmer(taskAccNorm ~ 1 + (1|ID), data = VDR_long[VDR_long$time!="V0",])
summary(m1.VDRACC)
exp(.5*diff(anova(m0.VDRACC,m1.VDRACC)$BIC)) #Bayes Factor 01


#####
#physio main effects
#####
fbands = c("delta","theta","alpha","beta","total")

colsel = c("ID","stimulation",colnames(SDR_all)[grepl("TF",colnames(SDR_all)) & !grepl("Norm",colnames(SDR_all))])

tf_sdr = reshape2::melt(SDR_long[,colsel],id.vars=c("ID","stimulation"))
tf_vdr = reshape2::melt(VDR_long[,colsel],id.vars=c("ID","stimulation"))

tf_sdr$variable = as.character(tf_sdr$variable)
tf_vdr$variable = as.character(tf_vdr$variable)
for (fband in fbands){
tf_sdr$variable[grepl(fband,tf_sdr$variable)] = fband
tf_vdr$variable[grepl(fband,tf_vdr$variable)] = fband
}
tf_sdr$variable = as.factor(tf_sdr$variable)
tf_sdr$variable = ordered(tf_sdr$variable, levels=fbands)
tf_vdr$variable = as.factor(tf_vdr$variable)
tf_vdr$variable = ordered(tf_vdr$variable, levels=fbands)

tfsdr=ggplot(tf_sdr, aes(x = variable, y = value, color=stimulation)) + stat_summary(position=position_dodge(width=.3), fun.data = "mean_cl_boot") + #geom_jitter()+
  theme_minimal() + theme(panel.grid = element_blank())+ ggtitle("SDR") + scale_color_manual(breaks=c("off","oDBS", "dDBS"), values=c("#21908c", "#440154", "#fde725"))+
  ylab('Spectral response [dB]') + xlab('Frequency band')+geom_hline(yintercept=0)


tfvdr=ggplot(tf_vdr, aes(x = variable, y = value, color=stimulation)) + stat_summary(position=position_dodge(width=.3), fun.data = "mean_cl_boot") + #geom_jitter()+
  theme_minimal() + theme(panel.grid = element_blank())+ ggtitle("VDR") + scale_color_manual(breaks=c("off","oDBS", "dDBS"), values=c("#21908c", "#440154", "#fde725"))+
  ylab('Spectral response [dB]') + xlab('Frequency band')+geom_hline(yintercept=0)

tfsdr/tfvdr
ggsave(file.path(figurePath, "fig4_spectral_maineffect.png"), height = 4,width = 7)

#models for spectral main effect
m1.delta.VDR=lmer(TFdelta..1.3. ~ stimulation + (1|ID), data = VDR_long)
summary(m1.delta.VDR)
m1.theta.VDR=lmer(TFtheta..4.8. ~ stimulation + (1|ID), data = VDR_long)
summary(m1.theta.VDR)
m1.alpha.VDR=lmer(TFalpha..9.14. ~ stimulation + (1|ID), data = VDR_long)
summary(m1.alpha.VDR)
m1.beta.VDR=lmer(TFbeta..15.30. ~ stimulation + (1|ID), data = VDR_long)
summary(m1.beta.VDR)

m1.delta.SDR=lmer(TFdelta..1.3. ~ stimulation + (1|ID), data = SDR_long)
summary(m1.delta.SDR)
m1.theta.SDR=lmer(TFtheta..4.8. ~ stimulation + (1|ID), data = SDR_long)
summary(m1.theta.SDR)
m1.alpha.SDR=lmer(TFalpha..9.14. ~ stimulation + (1|ID), data = SDR_long)
summary(m1.alpha.SDR)
m1.beta.SDR=lmer(TFbeta..15.30. ~ stimulation + (1|ID), data = SDR_long)
summary(m1.beta.SDR)

#test for stimulation effect on spectral response
m0.delta.VDR=lmer(TFdelta..1.3. ~ 1 + (1|ID), data = VDR_long)
anova(m1.delta.VDR)
exp(.5*diff(anova(m1.delta.VDR, m0.delta.VDR)$BIC)) #Bayes Factor 01

m0.theta.VDR=lmer(TFtheta..4.8. ~ 1 + (1|ID), data = VDR_long)
anova(m1.theta.VDR)
exp(.5*diff(anova(m1.theta.VDR, m0.theta.VDR)$BIC)) #Bayes Factor 01

m0.alpha.VDR=lmer(TFalpha..9.14. ~ 1 + (1|ID), data = VDR_long)
exp(.5*diff(anova(m1.alpha.VDR, m0.alpha.VDR)$BIC)) #Bayes Factor 01

m0.beta.VDR=lmer(TFbeta..15.30. ~ 1 + (1|ID), data = VDR_long)
exp(.5*diff(anova(m1.beta.VDR, m0.beta.VDR)$BIC)) #Bayes Factor 01

m0.delta.SDR=lmer(TFdelta..1.3. ~ 1 + (1|ID), data = SDR_long)
exp(.5*diff(anova(m1.delta.SDR, m0.delta.SDR)$BIC)) #Bayes Factor 01

m0.theta.SDR=lmer(TFtheta..4.8. ~ 1 + (1|ID), data = SDR_long)
exp(.5*diff(anova(m1.theta.SDR, m0.theta.SDR)$BIC)) #Bayes Factor 01

m0.alpha.SDR=lmer(TFalpha..9.14. ~ 1 + (1|ID), data = SDR_long)
exp(.5*diff(anova(m1.alpha.SDR, m0.alpha.SDR)$BIC)) #Bayes Factor 01

m0.beta.SDR=lmer(TFbeta..15.30. ~ 1 + (1|ID), data = SDR_long)
anova(m1.beta.SDR)
exp(.5*diff(anova(m1.beta.SDR, m0.beta.SDR)$BIC)) #Bayes Factor 01




##spectral-behavioral correlation
#TODO rename columns
#TODO generate for all freq bands

#task.acc and task.rt by freq and task
blcorr_sdr = merge(SDR_long[SDR_long$time=="V0",colsel],SDR_long[SDR_long$time !="V0",],by="ID")
blcorr_vdr = merge(VDR_long[VDR_long$time=="V0",colsel],VDR_long[VDR_long$time !="V0",],by="ID")

corplot_sdr=ggplot(blcorr_sdr) +  stat_smooth(method = "lm",se=F)+geom_point(aes(color=stimulation.y)) +
  theme_minimal() + theme(panel.grid = element_blank(), legend.title = element_blank()) +  guides(col='none') 
corplot_vdr=ggplot(blcorr_vdr) +  stat_smooth(method = "lm",se=F)+geom_point(aes(color=stimulation.y)) +
  theme_minimal() + theme(panel.grid = element_blank(), legend.title = element_blank()) +  guides(col='none') 

#baseline ephys / SDR Acc
tmp=cor.test(blcorr_sdr$TFdelta..1.3..x, blcorr_sdr$taskAccNorm, method='pearson')
title_ = sprintf("SDR / delta\nr=%.3f; p=%.3f", tmp$estimate, tmp$p.value)
delta_SDR_acc = corplot_sdr + aes(x = TFdelta..1.3..x, y = taskAccNorm)+ ggtitle(title_)+ylab('task accuracy\n(change to baseline)')+ xlab('spectral response at baseline [dB]')

tmp=cor.test(blcorr_sdr$TFtheta..4.8..x, blcorr_sdr$taskAccNorm, method='pearson')
title_ = sprintf("SDR / theta\nr=%.3f; p=%.3f", tmp$estimate, tmp$p.value)
theta_SDR_acc = corplot_sdr + aes(x = TFtheta..4.8..x, y = taskAccNorm)+ ggtitle(title_)+ylab('task accuracy\n(change to baseline)')+ xlab('spectral response at baseline [dB]')

tmp=cor.test(blcorr_sdr$TFalpha..9.14..x, blcorr_sdr$taskAccNorm, method='pearson')
title_ = sprintf("SDR / alpha\nr=%.3f; p=%.3f", tmp$estimate, tmp$p.value)
alpha_SDR_acc = corplot_sdr + aes(x = TFalpha..9.14..x, y = taskAccNorm)+ ggtitle(title_)+ylab('task accuracy\n(change to baseline)')+ xlab('spectral response at baseline [dB]')

tmp=cor.test(blcorr_sdr$TFbeta..15.30..x, blcorr_sdr$taskAccNorm, method='pearson')
title_ = sprintf("SDR / beta\nr=%.3f; p=%.3f", tmp$estimate, tmp$p.value)
beta_SDR_acc = corplot_sdr + aes(x = TFbeta..15.30..x, y = taskAccNorm)+ ggtitle(title_)+ylab('task accuracy\n(change to baseline)')+ xlab('spectral response at baseline [dB]')


#baseline ephys / SDR RT
tmp=cor.test(blcorr_sdr$TFdelta..1.3..x, blcorr_sdr$taskRTNorm, method='pearson')
title_ = sprintf("SDR / delta\nr=%.3f; p=%.3f", tmp$estimate, tmp$p.value)
delta_SDR_rt = corplot_sdr + aes(x = TFdelta..1.3..x, y = taskRTNorm)+ ggtitle(title_)+ylab('task RT [s]\n(change to baseline)')+ xlab('spectral response at baseline [dB]')

tmp=cor.test(blcorr_sdr$TFtheta..4.8..x, blcorr_sdr$taskRTNorm, method='pearson')
title_ = sprintf("SDR / theta\nr=%.3f; p=%.3f", tmp$estimate, tmp$p.value)
theta_SDR_rt = corplot_sdr + aes(x = TFtheta..4.8..x, y = taskRTNorm)+ ggtitle(title_)+ylab('task RT [s]\n(change to baseline)')+ xlab('spectral response at baseline [dB]')

tmp=cor.test(blcorr_sdr$TFalpha..9.14..x, blcorr_sdr$taskRTNorm, method='pearson')
title_ = sprintf("SDR / alpha\nr=%.3f; p=%.3f", tmp$estimate, tmp$p.value)
alpha_SDR_rt = corplot_sdr + aes(x = TFalpha..9.14..x, y = taskRTNorm)+ ggtitle(title_)+ylab('task RT [s]\n(change to baseline)')+ xlab('spectral response at baseline [dB]')

tmp=cor.test(blcorr_sdr$TFbeta..15.30..x, blcorr_sdr$taskRTNorm, method='pearson')
title_ = sprintf("SDR / beta\nr=%.3f; p=%.3f", tmp$estimate, tmp$p.value)
beta_SDR_rt = corplot_sdr + aes(x = TFbeta..15.30..x, y = taskRTNorm)+ ggtitle(title_)+ylab('task RT [s]\n(change to baseline)')+ xlab('spectral response at baseline [dB]')


(delta_SDR_acc | theta_SDR_acc | alpha_SDR_acc | beta_SDR_acc)/(delta_SDR_rt | theta_SDR_rt | alpha_SDR_rt | beta_SDR_rt)
ggsave(file.path(figurePath, 'blcorrs_SDR.svg'),width = 14,height = 5)



#baseline ephys / VDR Acc
tmp=cor.test(blcorr_vdr$TFdelta..1.3..x, blcorr_vdr$taskAccNorm, method='pearson')
title_ = sprintf("VDR / delta\nr=%.3f; p=%.3f", tmp$estimate, tmp$p.value)
delta_VDR_acc = corplot_vdr + aes(x = TFdelta..1.3..x, y = taskAccNorm)+ ggtitle(title_)+ylab('task accuracy\n(change to baseline)')+ xlab('spectral response at baseline [dB]')

tmp=cor.test(blcorr_vdr$TFtheta..4.8..x, blcorr_vdr$taskAccNorm, method='pearson')
title_ = sprintf("VDR / theta\nr=%.3f; p=%.3f", tmp$estimate, tmp$p.value)
theta_VDR_acc = corplot_vdr + aes(x = TFtheta..4.8..x, y = taskAccNorm)+ ggtitle(title_)+ylab('task accuracy\n(change to baseline)')+ xlab('spectral response at baseline [dB]')

tmp=cor.test(blcorr_vdr$TFalpha..9.14..x, blcorr_vdr$taskAccNorm, method='pearson')
title_ = sprintf("VDR / alpha\nr=%.3f; p=%.3f", tmp$estimate, tmp$p.value)
alpha_VDR_acc = corplot_vdr + aes(x = TFalpha..9.14..x, y = taskAccNorm)+ ggtitle(title_)+ylab('task accuracy\n(change to baseline)')+ xlab('spectral response at baseline [dB]')

tmp=cor.test(blcorr_vdr$TFbeta..15.30..x, blcorr_vdr$taskAccNorm, method='pearson')
title_ = sprintf("VDR / beta\nr=%.3f; p=%.3f", tmp$estimate, tmp$p.value)
beta_VDR_acc = corplot_vdr + aes(x = TFbeta..15.30..x, y = taskAccNorm)+ ggtitle(title_)+ylab('task accuracy\n(change to baseline)')+ xlab('spectral response at baseline [dB]')


#baseline ephys / VDR RT
tmp=cor.test(blcorr_vdr$TFdelta..1.3..x, blcorr_vdr$taskRTNorm, method='pearson')
title_ = sprintf("VDR / delta\nr=%.3f; p=%.3f", tmp$estimate, tmp$p.value)
delta_VDR_rt = corplot_vdr + aes(x = TFdelta..1.3..x, y = taskRTNorm)+ ggtitle(title_)+ylab('task RT [s]\n(change to baseline)')+ xlab('spectral response at baseline [dB]')

tmp=cor.test(blcorr_vdr$TFtheta..4.8..x, blcorr_vdr$taskRTNorm, method='pearson')
title_ = sprintf("VDR / theta\nr=%.3f; p=%.3f", tmp$estimate, tmp$p.value)
theta_VDR_rt = corplot_vdr + aes(x = TFtheta..4.8..x, y = taskRTNorm)+ ggtitle(title_)+ylab('task RT [s]\n(change to baseline)')+ xlab('spectral response at baseline [dB]')

tmp=cor.test(blcorr_vdr$TFalpha..9.14..x, blcorr_vdr$taskRTNorm, method='pearson')
title_ = sprintf("VDR / alpha\nr=%.3f; p=%.3f", tmp$estimate, tmp$p.value)
alpha_VDR_rt = corplot_vdr + aes(x = TFalpha..9.14..x, y = taskRTNorm)+ ggtitle(title_)+ylab('task RT [s]\n(change to baseline)')+ xlab('spectral response at baseline [dB]')

tmp=cor.test(blcorr_vdr$TFbeta..15.30..x, blcorr_vdr$taskRTNorm, method='pearson')
title_ = sprintf("VDR / beta\nr=%.3f; p=%.3f", tmp$estimate, tmp$p.value)
beta_VDR_rt = corplot_vdr + aes(x = TFbeta..15.30..x, y = taskRTNorm)+ ggtitle(title_)+ylab('task RT [s]\n(change to baseline)')+ xlab('spectral response at baseline [dB]')


(delta_VDR_acc | theta_VDR_acc | alpha_VDR_acc | beta_VDR_acc)/(delta_VDR_rt | theta_VDR_rt | alpha_VDR_rt | beta_VDR_rt)
ggsave(file.path(figurePath, 'blcorrs_VDR.svg'),width = 14,height = 5)

