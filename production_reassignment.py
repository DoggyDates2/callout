// Google Apps Script to watch M1 cell and trigger GitHub Actions
// 1. Go to script.google.com
// 2. Create new project
// 3. Replace Code.gs with this script
// 4. Set up trigger for onChange

// Configuration - UPDATE THESE
const CONFIG = {
  // Your GitHub repository info
  GITHUB_OWNER: 'your-username',
  GITHUB_REPO: 'your-repo-name', 
  GITHUB_TOKEN: 'your-github-token', // GitHub Personal Access Token
  
  // Your Google Sheets info - ACTUAL VALUES
  SPREADSHEET_ID: '1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0',
  SHEET_NAME: 'New districts Map 8', // or whatever your sheet is called
  WATCH_CELL: 'M1'
};

// Store the last known value of M1
let lastM1Value = null;

function onEdit(e) {
  // Check if the edited cell is M1
  const range = e.range;
  const sheet = range.getSheet();
  
  if (sheet.getName() === CONFIG.SHEET_NAME && range.getA1Notation() === CONFIG.WATCH_CELL) {
    const newValue = range.getValue();
    
    Logger.log(`M1 changed to: ${newValue}`);
    
    // Only trigger if value actually changed
    if (newValue !== lastM1Value) {
      lastM1Value = newValue;
      triggerGitHubAction(newValue);
    }
  }
}

function onOpen() {
  // Check M1 value when sheet opens to initialize
  const sheet = SpreadsheetApp.openById(CONFIG.SPREADSHEET_ID).getSheetByName(CONFIG.SHEET_NAME);
  const currentValue = sheet.getRange(CONFIG.WATCH_CELL).getValue();
  lastM1Value = currentValue;
  
  Logger.log(`Initial M1 value: ${currentValue}`);
}

function triggerGitHubAction(m1Value) {
  try {
    const url = `https://api.github.com/repos/${CONFIG.GITHUB_OWNER}/${CONFIG.GITHUB_REPO}/dispatches`;
    
    const payload = {
      event_type: 'm1-changed',
      client_payload: {
        m1_value: m1Value,
        timestamp: new Date().toISOString(),
        spreadsheet_id: CONFIG.SPREADSHEET_ID
      }
    };
    
    const options = {
      method: 'POST',
      headers: {
        'Authorization': `token ${CONFIG.GITHUB_TOKEN}`,
        'Accept': 'application/vnd.github.v3+json',
        'Content-Type': 'application/json'
      },
      payload: JSON.stringify(payload)
    };
    
    const response = UrlFetchApp.fetch(url, options);
    
    if (response.getResponseCode() === 204) {
      Logger.log('✅ Successfully triggered GitHub Action');
      
      // Optional: Show toast notification in Google Sheets
      SpreadsheetApp.getActiveSpreadsheet().toast(
        'Dog reassignment system triggered!', 
        'M1 Changed', 
        3
      );
    } else {
      Logger.log(`❌ GitHub API error: ${response.getResponseCode()} - ${response.getContentText()}`);
    }
    
  } catch (error) {
    Logger.log(`❌ Error triggering GitHub Action: ${error}`);
  }
}

// Test function to manually trigger (for testing)
function testTrigger() {
  triggerGitHubAction('TEST_VALUE');
}
