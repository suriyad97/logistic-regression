param location string = 'eastus'

resource resourceGroup 'Microsoft.Resources/resourceGroups@2023-01-01' = {
  name: 'rg-titanic-ml-${uniqueString(subscription().id)}'
  location: location
}

module mlInfrastructure './main.bicep' = {
  scope: resourceGroup
  name: 'mlInfrastructure'
  params: {
    location: location
    workspaceName: 'titanic-ml-ws'
    storageAccountName: 'titanicml'
    keyVaultName: 'titanic-kv'
    appInsightsName: 'titanic-insights'
  }
}

output resourceGroupName string = resourceGroup.name
output workspaceName string = mlInfrastructure.outputs.workspaceName
