"""Map strings to classes."""
from ase.models import (
    SKLearnModel,
    RadialBNN, TinyRadialBNN, ResNet18,
    WideResNet, ResNet18Ensemble, ResNet34Ensemble, ResNet50Ensemble,
    ResNet101Ensemble, WideResNetEnsemble)
from ase.datasets import (
    OpenMLDataset,
    MNISTDataset, FashionMNISTDataset,
    Cifar10Dataset, Cifar100Dataset)
from ase.acquisition import (
    RandomAcquisition, TrueLossAcquisition,
    ClassifierAcquisitionEntropy,
    SelfSurrogateAcquisitionEntropy,
    SelfSurrogateAcquisitionSurrogateEntropy,
    SelfSurrogateAcquisitionSurrogateMutualInformation,
    AnySurrogateAcquisitionEntropy,
    SelfSurrogateAcquisitionSurrogateMI,
    SelfSurrogateAcquisitionSurrogateWeightedBALD2,
    )
from ase.loss import (
    BalancedAccuracy, BalancedAccuracyLoss,
    SELoss, MSELoss, RMSELoss, CrossEntropyLoss, AccuracyLoss, YIsLoss)

from ase.risk_estimators import (
    BiasedRiskEstimator, NaiveUnbiasedRiskEstimator,
    FancyUnbiasedRiskEstimator, TrueRiskEstimator,
    TrueUnseenRiskEstimator,
    QuadratureRiskEstimator,
    ExactExpectedRiskEstimator,
    FullSurrogateASMC,
    )


model = dict(
    SKLearnModel=SKLearnModel,
    RadialBNN=RadialBNN,
    TinyRadialBNN=TinyRadialBNN,
    ResNet18=ResNet18,
    WideResNet=WideResNet,
    ResNet18Ensemble=ResNet18Ensemble,
    ResNet34Ensemble=ResNet34Ensemble,
    ResNet50Ensemble=ResNet50Ensemble,
    ResNet101Ensemble=ResNet101Ensemble,
    WideResNetEnsemble=WideResNetEnsemble,
)

dataset = dict(
    OpenMLDataset=OpenMLDataset,
    MNISTDataset=MNISTDataset,
    FashionMNISTDataset=FashionMNISTDataset,
    Cifar10Dataset=Cifar10Dataset,
    Cifar100Dataset=Cifar100Dataset,
)

acquisition = dict(
    RandomAcquisition=RandomAcquisition,
    TrueLossAcquisition=TrueLossAcquisition,
    ClassifierAcquisitionEntropy=ClassifierAcquisitionEntropy,
    SelfSurrogateAcquisitionEntropy=SelfSurrogateAcquisitionEntropy,
    SelfSurrogateAcquisitionSurrogateEntropy=(
        SelfSurrogateAcquisitionSurrogateEntropy),
    SelfSurrogateAcquisitionSurrogateMutualInformation=(
        SelfSurrogateAcquisitionSurrogateMutualInformation),
    AnySurrogateAcquisitionEntropy=AnySurrogateAcquisitionEntropy,
    SelfSurrogateAcquisitionSurrogateMI=SelfSurrogateAcquisitionSurrogateMI,
    SelfSurrogateAcquisitionSurrogateWeightedBALD2=(
        SelfSurrogateAcquisitionSurrogateWeightedBALD2
    ),
)

loss = dict(
    BalancedAccuracy=BalancedAccuracy,
    BalancedAccuracyLoss=BalancedAccuracyLoss,
    SELoss=SELoss,
    MSELoss=MSELoss,
    RMSELoss=RMSELoss,
    CrossEntropyLoss=CrossEntropyLoss,
    AccuracyLoss=AccuracyLoss,
    YIsLoss=YIsLoss,
)

risk_estimator = dict(
    TrueRiskEstimator=TrueRiskEstimator,
    BiasedRiskEstimator=BiasedRiskEstimator,
    NaiveUnbiasedRiskEstimator=NaiveUnbiasedRiskEstimator,
    FancyUnbiasedRiskEstimator=FancyUnbiasedRiskEstimator,
    TrueUnseenRiskEstimator=TrueUnseenRiskEstimator,
    QuadratureRiskEstimator=QuadratureRiskEstimator,
    ExactExpectedRiskEstimator=ExactExpectedRiskEstimator,
    FullSurrogateASMC=FullSurrogateASMC,
)
