import os,time
from myutils.dataset import *
from myutils.models import *
from myutils.tracking import get_callbacks_exp

def experimental_protocol_multi_run(enhanced_dir,run_dir ,datasets, models, data_generators, scale=1, batch_size=64, epochs=16, fit_verbosity=0, tag_id='lastest'):
   
    # ---- Logs and models dir
    #
    os.makedirs(f'{run_dir}/logs_{tag_id}',   mode=0o750, exist_ok=True)
    os.makedirs(f'{run_dir}/models_{tag_id}', mode=0o750, exist_ok=True)
    
    # Meta data
    csv_metadtat_output='enhanced_dir,run_dir,scale,batch_size,epochs,tag_id\n'
    csv_metadtat_output+=f'{enhanced_dir},{run_dir},{scale},{batch_size},{epochs},{tag_id}\n'

    ## Report
    csv_report_output='Dataset,DatasetSize,Model,Datagen,Duration,Accuracy\n'


    # ---- For each dataset
    for d_name in datasets:
        fidle.utils.subtitle(f"Dataset : {d_name}")
       
        # ---- Read dataset
        x_train,y_train, x_test,y_test,labels = read_dataset(enhanced_dir, d_name)
        d_size = dataset_size(enhanced_dir, d_name)
       
        
        # ---- Rescale
        x_train,y_train,x_test,y_test = fidle.utils.rescale_dataset(x_train,y_train,x_test,y_test, scale=scale)
        
        # ---- Get the shape
        (n,lx,ly,lz) = x_train.shape

        # ---- For each model
        for m_function in models :
            m_name = m_function[4:]

            print("    Run model {}  : ".format(m_name))
            # ---- get model
            try:
                # ---- get function by name
                m_function=globals()[m_function]
                model=m_function(lx,ly,lz)
               

                # ---- Compile it
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


                # ---- For each datagenerator
                for g_function in data_generators:
                   
                    g_name=g_function[9:] if g_function != None else 'none'

                    if g_function != None:
                        g_function=globals()[g_function]
                        datagen = g_function()
                    else:
                        datagen = None

                    print(f'          With Datagenerator: {g_name!="none"}')

                    # Callbacks

                    callbacks = get_callbacks_exp(run_dir,tag_id,d_name,m_name,g_name)
    
                    
                    start_time = time.time()

                    if datagen==None:
                        # ---- No data augmentation --------------------------------------
                        history = model.fit(x_train, y_train,
                                            batch_size      = batch_size,
                                            epochs          = epochs,
                                            verbose         = fit_verbosity,
                                            validation_data = (x_test, y_test),
                                            callbacks       = callbacks)
                    else:
                        # ---- Data augmentation ----------------------------------------
                        datagen.fit(x_train)
                        history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                                            steps_per_epoch = int(len(x_train)/batch_size),
                                            epochs          = epochs,
                                            verbose         = fit_verbosity,
                                            validation_data = (x_test, y_test),
                                            callbacks       = callbacks)
                        
                    # ---- Result
                    end_time = time.time()
                    duration = end_time-start_time
                    accuracy = max(history.history["val_accuracy"])*100
                    #
                    
                    # ---- Report
                    csv_report_output+=f'{d_name},{d_size},{m_name},{g_name},{duration},{accuracy}\n'

                    print(f"Accuracy={accuracy: 7.2f}    Duration={duration: 7.2f}")
            except:
                print('An error occured for :',m_name)
                print('-')
    return csv_report_output,csv_metadtat_output