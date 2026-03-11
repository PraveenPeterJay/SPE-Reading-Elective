pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Jenkins sudo test') {
            steps {
                sh '''
                    sudo apt-get update 2>&1 || true
                    sudo -l -U jenkins  # Check what jenkins can sudo
                    cat /etc/apt/sources.list
                    ls /etc/apt/sources.list.d/
                '''
            }
        }

        stage('Run Ansible Pipeline') {
            steps {
                sh '''
                    # Verify sudo works before ansible runs
                    sudo apt-get update -qq
                    ansible-playbook -i ansible/inventory.ini ansible/pipeline.yml -vvv
                '''
            }
        }
    }

    post {
        always {
            // This ensures results are saved even if a test or build fails
            echo "Archiving research data..."
            archiveArtifacts artifacts: '*.json, *.txt', allowEmptyArchive: true
        }
    }
}