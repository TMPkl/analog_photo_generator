const uploadContainer = document.getElementById('upload-container');
const imageUploadInput = document.getElementById('image-upload');

uploadContainer.addEventListener('click', () => {
    imageUploadInput.click();
});

document.getElementById('select-button').addEventListener('click', (e) => {
    e.stopPropagation();
    imageUploadInput.click();
});

// Handle drag-over effect
uploadContainer.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadContainer.classList.add('dragover');
});

uploadContainer.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadContainer.classList.remove('dragover');
});

uploadContainer.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadContainer.classList.remove('dragover');

    if (e.dataTransfer.files.length > 0) {
        imageUploadInput.files = e.dataTransfer.files;

        const file = e.dataTransfer.files[0];
        const reader = new FileReader();
        reader.onload = function (event) {
            document.getElementById('uploaded-image-preview').src = event.target.result;
            document.getElementById('upload-container').style.display = 'none';
            document.getElementById('uploaded-image-section').style.display = 'block';
            document.getElementById('tools-grid').style.display = 'block';
        };
        reader.readAsDataURL(file);
        
        uploadSelectedFile(file);
    }
});

// Still handle normal file input change
imageUploadInput.addEventListener('change', function (e) {
    if (e.target.files.length > 0) {
        const file = e.target.files[0];
        const reader = new FileReader();
        reader.onload = function (event) {
            document.getElementById('uploaded-image-preview').src = event.target.result;
            document.getElementById('upload-container').style.display = 'none';
            document.getElementById('uploaded-image-section').style.display = 'block';
            document.getElementById('tools-grid').style.display = 'block';

        };
        reader.readAsDataURL(file);
        
        uploadSelectedFile(file);
    }
});

document.getElementById('upload-another').addEventListener('click', function (e) {
    e.stopPropagation();
    imageUploadInput.value = ""; // Clear file input
    document.getElementById('upload-container').style.display = 'block';
    document.getElementById('uploaded-image-section').style.display = 'none';
    document.getElementById('tools-grid').style.display = 'none';
    document.getElementById('result-container').style.display = 'none';
});

document.querySelectorAll('.tool-card').forEach(card => {
    card.addEventListener('click', onCardClick);
});

const TOOL_INPUTS_CONFIG =
    {
        1:
        [
            {
                name: 'tool_input_scales',
                label: 'Noise Scales (space-separated floats)',
                initial_value: '1 0.2 0.4', 
                type: 'text',
                flag: '--scale'
            },
            {
                name: 'tool_input_intensity',
                label: 'Intensity (float)',
                initial_value: 0.4,
                type: 'number',
                flag: '--intensity'
            },
            {
                name: 'tool_input_grain_amplitude',
                label: 'Grain Amplitude (float)',
                initial_value: 0.18,
                type: 'number',
                flag: '--grain_amplitude'
            }
        ],
        2:
        [
            {
                name: 'tool_input_bright_threshold',
                label: 'Brightness Threshold (0-255)',
                type: 'number',
                options: {'min': 0, 'max': 255},
                initial_value: 243,
                flag: "--bright_threshold",
            },
            {
                name: 'tool_input_kernel_size',
                label: 'Halation Map Kernel Size',
                type: 'number',
                options: {'min': 0},
                initial_value: 55,
                flag: "--kernel_size",
            },
            {
                name: 'tool_input_sigma_x',
                label: 'Halation Map Sigma X',
                type: 'number',
                options: {'min': 0},
                initial_value: 40,
                flag: "--sigma_x",
            },
            {
                name: 'tool_input_delta_mode',
                label: 'Halation Map Delta Mode',
                type: 'select',
                options: {'Yes': 1, "No": 0},
                flag: "--delta_mode",
            },
            {
                name: 'tool_input_intensity',
                label: 'Halation Intensity',
                type: 'number',
                options: {'min': 0},
                initial_value: 0.1,
                flag: "--intensity",
            },
            
        ],
        3:
        [
            {
                name: "tool_input_lut",
                label: "LUT file to use",
                type: "select",
                options: {
                    "Reinhard all dataset": "1",
                    "OT color transfer form ": "2",
                },
                flag: "--lut",
            },
        ]
        // 4: [
        //     {
        //         name: 'tool_input_color_space',
        //         label: 'Resulting color space',
        //         type: 'select',
        //         options: {"Gray": "GRAY", "HSV": "HSV", "RGB": "BGR"},
        //         flag: "",
        //     }
        // ],
        // 2: [
        //     {
        //         name: 'tool_input_color_space',
        //         label: 'Resulting color space',
        //         type: 'select',
        //         options: {"Gray": "GRAY", "HSV": "HSV", "RGB": "BGR"},
        //         flag: "",
        //     }
        // ],
        // 6: [
        //     {
        //         name: 'tool_input_width',
        //         label: "Resulting width",
        //         type: 'number',
        //         options: {'min': 1},
        //         flag: "-w"
        //     },
        //     {
        //         name: 'tool_input_height',
        //         label: "Resulting height",
        //         type: 'number',
        //         options: {'min': 1},
        //         flag: "-H"
        //     },
        // ]
    }

function onCardClick()
{
    const toolId = this.getAttribute('data-toolid');
    const fileId = document.getElementById("uploaded-file-id").value;
    
    // If the tool requires extra input, show the specified form
    if (toolId in TOOL_INPUTS_CONFIG) {
        onAfterCardShowForm(toolId, fileId);
        return;
    }
    
    onRunTool(toolId, fileId);
}

function onAfterCardShowForm(toolId, fileId)
{
    // Check if the form already exists for this tool
    if(document.getElementById('tool-input-'+toolId)) {
        return;
    }
    
    const existingInputWrapper = document.querySelector('.tool-input-wrapper');
    if(existingInputWrapper) {
        existingInputWrapper.remove();
    }
    
    const formWrapper = document.createElement('div');
    formWrapper.classList.add('tool-input-wrapper');
    formWrapper.id = 'tool-input-'+toolId;
    
    // create the options and append them to the formWrapper
    for (let inputData of TOOL_INPUTS_CONFIG[toolId]) {
        let inputWrapper = document.createElement('div');
        inputWrapper.classList.add('form-group');
        
        let label = document.createElement('label');
        label.classList.add('form-label');
        label.setAttribute('for', inputData.name);
        label.innerHTML = inputData.label;
        
        inputWrapper.appendChild(label);
        
        switch (inputData.type) {
            case 'number':
                let inputNumber = document.createElement('input');
                inputNumber.classList.add('form-control');
                inputNumber.setAttribute('type', 'number');
                inputNumber.setAttribute('name', inputData.name);
                inputNumber.id = inputData.name;
                
                if(inputData.options && 'min' in inputData.options)
                    inputNumber.setAttribute('min', inputData.options.min);
                
                if(inputData.options && 'max' in inputData.options)
                    inputNumber.setAttribute('max', inputData.options.max);

                if(inputData.initial_value)
                    inputNumber.setAttribute('value', inputData.initial_value);
                
                inputWrapper.appendChild(inputNumber);
                break;
                
            case 'text':
            let inputText = document.createElement('input');
            inputText.classList.add('form-control');
            inputText.setAttribute('type', 'text');
            inputText.setAttribute('name', inputData.name);
            if(inputData.initial_value)
                inputText.setAttribute('value', inputData.initial_value);
            inputText.id = inputData.name;
            
            inputWrapper.appendChild(inputText);
            break;
                
            case 'select':
                let inputSelect = document.createElement('select');
                inputSelect.classList.add('form-control');
                inputSelect.name = inputData.name;
                inputSelect.id = inputData.name;
                
                for (let inputOptionKey in inputData.options) {
                    let inputOptionValue = inputData.options[inputOptionKey];
                    
                    let inputOption = document.createElement('option');
                    inputOption.value = inputOptionValue;
                    inputOption.innerHTML = inputOptionKey;
                    
                    inputSelect.appendChild(inputOption);
                }
                
                inputWrapper.appendChild(inputSelect);
        }
        
        formWrapper.appendChild(inputWrapper);
        
        
    }

    let submitButton = document.createElement('button');
    submitButton.classList.add('btn');
    submitButton.classList.add('btn-primary');
    submitButton.innerHTML = 'Submit';
    submitButton.addEventListener('click', ()=>onRunTool(toolId, fileId));
    formWrapper.appendChild(submitButton);
    
    formWrapper.classList.add('mt-2');
    
    document.getElementById('tools-grid').after(formWrapper);
    
}

function hideToolForm()
{
    const existingInputWrapper = document.querySelector('.tool-input-wrapper');
    if(existingInputWrapper) {
        existingInputWrapper.remove();
    }
}

function validateToolInputs(toolId)
{
    let error = false;
    
    if(toolId in TOOL_INPUTS_CONFIG)
    {
        document.querySelectorAll('.tool-input-wrapper .text-danger').forEach(span => span.remove());

        for (let input of TOOL_INPUTS_CONFIG[toolId]) {
            const inputElement = document.getElementById(input.name);
            if(!inputElement || inputElement.value.trim() === "")
            {
                error = true;
                    
                const errorSpan = document.createElement('span');
                errorSpan.classList.add('text-danger');
                errorSpan.textContent = `${input.label} may not be empty.`;
                
                if(inputElement)
                    inputElement.parentElement.appendChild(errorSpan);
                else 
                    document.querySelector('.tool-input-wrapper').appendChild(errorSpan);
            }
            else
            {
                if(input.name === 'tool_input_width' || input.name === 'tool_input_height')
                {
                    if(parseInt(inputElement.value) <= 0)
                    {
                        error = true;

                        const errorSpan = document.createElement('span');
                        errorSpan.classList.add('text-danger');
                        errorSpan.textContent = `${input.label} may not be 0.`;

                        if(inputElement)
                            inputElement.parentElement.appendChild(errorSpan);
                        else
                            document.querySelector('.tool-input-wrapper').appendChild(errorSpan);
                    }
                }
            }
            
        }
    }
    
    return !error;
}

function createToolOptions(toolId)
{
    if(!(toolId in TOOL_INPUTS_CONFIG))
        return {};

    let options = {};
    for (let input of TOOL_INPUTS_CONFIG[toolId]) {
        const inputElement = document.getElementById(input.name);
        options[input.name] = input.flag + " " + inputElement.value;
    }
    
    return options;
}

function showDetectedText(text)
{
    let textContainer= document.getElementById('detected-text');
    
    if(!textContainer)
    {
        textContainer = document.createElement('textarea');
        textContainer.id = 'detected-text';
        textContainer.classList.add('form-control');
        textContainer.classList.add('mb-3');
        textContainer.rows = 5;
        textContainer.readOnly = true;
        
        const resultContainer = document.getElementById('result-container');
        resultContainer.insertBefore(textContainer, document.getElementById('result-image'));
    }
    
    textContainer.value = text;
}

function hideDetectedText()
{
    let textContainer= document.getElementById('detected-text');
    
    if(textContainer)
        textContainer.remove();
}

function onRunTool(toolId, fileId)
{
    if(!validateToolInputs(toolId))
    {
        return;
    }
    
    // Disable buttons
    document.querySelectorAll('.tool-card').forEach(c => c.classList.add('disabled'));
    // Show loading spinner
    document.getElementById('result-container').style.display = 'block';
    document.getElementById('loading-indicator').style.display = 'block';
    document.getElementById('result-image').style.display = 'none';
    document.getElementById('download-result').style.display = 'none';
    hideError();

    const formData = new FormData();
    formData.append('file', fileId);
    formData.append('toolId', toolId); 
    formData.append('options', JSON.stringify(createToolOptions(toolId)));
    
    hideToolForm();
    hideDetectedText();

    fetch('/Tools/ProcessImage', {
        method: 'POST',
        body: formData
    })
        .then(async res => {
            const text = await res.text();

            if (!res.ok) {
                throw new Error(text); // Contains plain error string like "The output file was not created."
            }

            try {
                return JSON.parse(text); // Safely parse only if it's valid JSON
            } catch (e) {
                throw new Error("Invalid JSON response from server.");
            }
        })
        .then(data => {
            if (data.success) {
                document.getElementById('result-image').src = data.resultUrl;
                document.getElementById('result-image').style.display = 'block';

                document.getElementById('download-result').href = data.resultUrl;
                document.getElementById('download-result').setAttribute('download', data.downloadName);
                document.getElementById('download-result').style.display = 'inline-block';

                document.getElementById('loading-indicator').style.display = 'none';

                if (data.text) {
                    showDetectedText(data.text);
                }
            } else {
                showError('Processing failed. Please try again.');
            }

        }).catch(async err => {
        console.error(err);

        // Try to extract and show the error message from the response
        if (err instanceof Response) {
            try {
                const errorData = await err.json(); 
                showError(errorData.message || 'An error occurred.');
            } catch {
                const errorText = await err.text(); 
                showError(errorText || 'An unknown server error occurred.');
            }
        } else {
            showError(err.message || 'An unexpected error occurred while processing the image.');
        }
    })
        .finally(() => {
            document.getElementById('loading-indicator').style.display = 'none';
            document.querySelectorAll('.tool-card').forEach(c => c.classList.remove('disabled'));

            const resultSection = document.getElementById('result-container');
            if (resultSection) {
                resultSection.scrollIntoView({ behavior: 'smooth' , block: 'end'});
            }
        });
}

function uploadSelectedFile(file)
{
    const formData = new FormData();
    formData.append('file', file);
    
    fetch('/Tools/SaveUploadedFile', {
        method: 'POST',
        body: formData
    })
        .then(async res => {
            const text = await res.text();

            if (!res.ok) {
                throw new Error(text); 
            }

            try {
                return JSON.parse(text);
            } catch (e) {
                throw new Error("Invalid JSON response from server.");
            }
        })
        .then(data =>
        {
            if(data.success)
            {
                const fileId = data.fileId;
                
                const url = new URL(window.location.href);
                url.searchParams.set("file", fileId);
                
                history.replaceState(null, "", url);
                
                document.getElementById('uploaded-file-id').value = fileId;
                hideError();
            }
            else
            {
                document.getElementById('image-upload').value = "";
                showError(data.message || "An error occurred while uploading the file.");
                document.getElementById('upload-another').click();
            }
        })
        .catch(err => {
            document.getElementById('image-upload').value = "";
            showError(err || "An unexpected error occurred during upload.");
            document.getElementById('upload-another').click();
        });
}

function hideError()
{
    let errorContainer = document.getElementById('error-message');
    if(errorContainer) 
        errorContainer.remove();
}
function showError(message) {
    let errorContainer = document.getElementById('error-message');
    if (!errorContainer) {
        errorContainer = document.createElement('div');
        errorContainer.id = 'error-message';
        errorContainer.className = 'alert alert-danger mt-3';
        const content = document.getElementById('content');
        const resultContainer = document.getElementById('result-container');
        if (content && resultContainer) {
            content.insertBefore(errorContainer, resultContainer);
        }
    }

    errorContainer.textContent = message;
    errorContainer.style.display = 'block';
}


