from configuration import Configuration
from models.pyramid_net import PyramidNet
from models.uheight_net import UHeightNet

if __name__ == '__main__':
    configuration = Configuration(epochs=250,
                                  learning_rate=1e-5,
                                  batch_size=4,
                                  use_mixed_precision=False,
                                  min_height=0.0,
                                  max_height=180.0,
                                  training_dataset_limit=1,
                                  validation_dataset_limit=1,
                                  testing_dataset_limit=11)

    model = PyramidNet(configuration)
    trainer = configuration.create_trainer(model)

    print(configuration)
    model.summary()

    trainer.fit(model=model, train_dataloaders=configuration.training_data_loader,
                val_dataloaders=configuration.validation_data_loader)
