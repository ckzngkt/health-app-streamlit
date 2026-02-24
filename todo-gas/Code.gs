/**
 * Gemini ToDo App — Google Apps Script backend
 *
 * 【セットアップ手順】
 * 1. Google Apps Script (script.google.com) で新規プロジェクトを作成
 * 2. このファイルの内容を Code.gs に貼り付け、Index.html も同様に追加
 * 3. スプレッドシートを新規作成し、そのIDをコピー
 * 4. GAS の「プロジェクトの設定」→「スクリプトプロパティ」に以下を追加:
 *      SPREADSHEET_ID  : コピーしたスプレッドシートID
 *      GEMINI_API_KEY  : Google AI Studio (aistudio.google.com) で取得したAPIキー
 * 5. 「デプロイ」→「新しいデプロイ」→ 種類: ウェブアプリ
 *      実行ユーザー: 自分  / アクセスできるユーザー: 全員
 * 6. 表示されたURLがアプリのURLです
 */

// ─────────────────────────────────────────
// Constants
// ─────────────────────────────────────────
const SPREADSHEET_ID = PropertiesService.getScriptProperties().getProperty('SPREADSHEET_ID');
const TODO_SHEET = 'todos';
const GEMINI_MODEL = 'gemini-2.0-flash';

// ─────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────
function doGet() {
  return HtmlService.createHtmlOutputFromFile('Index')
    .setTitle('Gemini ToDo')
    .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL);
}

// ─────────────────────────────────────────
// Sheet helper
// ─────────────────────────────────────────
function getSheet_() {
  const ss = SpreadsheetApp.openById(SPREADSHEET_ID);
  let sheet = ss.getSheetByName(TODO_SHEET);
  if (!sheet) {
    sheet = ss.insertSheet(TODO_SHEET);
    sheet.appendRow(['id', 'title', 'notes', 'dueDate', 'priority', 'category', 'completed', 'completedAt', 'createdAt']);
    sheet.setFrozenRows(1);
    sheet.getRange('A1:I1').setFontWeight('bold');
  }
  return sheet;
}

// ─────────────────────────────────────────
// CRUD — called via google.script.run
// ─────────────────────────────────────────

/** 全ToDoを取得（作成日の降順） */
function getTodos() {
  const sheet = getSheet_();
  const data = sheet.getDataRange().getValues();
  if (data.length <= 1) return [];

  const headers = data[0];
  return data.slice(1)
    .map(row => {
      const todo = {};
      headers.forEach((h, i) => { todo[h] = row[i]; });
      return todo;
    })
    .filter(t => t.id)
    .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
}

/** ToDoを追加 */
function addTodo(todo) {
  const sheet = getSheet_();
  const id = Utilities.getUuid();
  const now = new Date().toISOString();

  sheet.appendRow([
    id,
    (todo.title  || '').trim(),
    (todo.notes  || '').trim(),
    todo.dueDate  || '',
    todo.priority || 'medium',
    (todo.category || '').trim(),
    false,
    '',
    now
  ]);

  return { id, title: todo.title, notes: todo.notes || '', dueDate: todo.dueDate || '',
           priority: todo.priority || 'medium', category: todo.category || '',
           completed: false, completedAt: '', createdAt: now };
}

/** 完了/未完了を切り替え */
function toggleTodo(id) {
  const sheet = getSheet_();
  const data  = sheet.getDataRange().getValues();
  const heads = data[0];
  const idCol  = heads.indexOf('id');
  const doneCol = heads.indexOf('completed');
  const doneAtCol = heads.indexOf('completedAt');

  for (let i = 1; i < data.length; i++) {
    if (String(data[i][idCol]) === String(id)) {
      const completed = !data[i][doneCol];
      sheet.getRange(i + 1, doneCol   + 1).setValue(completed);
      sheet.getRange(i + 1, doneAtCol + 1).setValue(completed ? new Date().toISOString() : '');
      return { success: true, completed };
    }
  }
  throw new Error('Todo not found: ' + id);
}

/** ToDoを削除 */
function deleteTodo(id) {
  const sheet = getSheet_();
  const data  = sheet.getDataRange().getValues();
  const idCol = data[0].indexOf('id');

  for (let i = 1; i < data.length; i++) {
    if (String(data[i][idCol]) === String(id)) {
      sheet.deleteRow(i + 1);
      return { success: true };
    }
  }
  throw new Error('Todo not found: ' + id);
}

// ─────────────────────────────────────────
// Gemini integration
// ─────────────────────────────────────────

/**
 * 自然言語テキストをGeminiで解析してToDo構造体に変換する
 * @param {string} text  ユーザーの自然言語入力
 * @returns {{ title, notes, dueDate, priority, category }}
 */
function parseWithGemini(text) {
  const apiKey = PropertiesService.getScriptProperties().getProperty('GEMINI_API_KEY');
  if (!apiKey) throw new Error('スクリプトプロパティに GEMINI_API_KEY が設定されていません');

  const today = Utilities.formatDate(new Date(), 'Asia/Tokyo', 'yyyy-MM-dd');
  const url   = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${apiKey}`;

  const prompt = `あなたはToDoアプリのアシスタントです。ユーザーの入力からToDoタスクを1件作成してください。
今日の日付: ${today}

以下のJSONフォーマットのみ返してください（余計な説明・コードブロック不要）:
{
  "title":    "タスクのタイトル（端的に）",
  "notes":    "補足メモ（なければ空文字）",
  "dueDate":  "YYYY-MM-DD形式（日時が含まれる場合のみ、なければ空文字）",
  "priority": "high または medium または low",
  "category": "仕事 / 買い物 / 家事 / 勉強 / プライベート など（なければ空文字）"
}

ユーザー入力: ${text}`;

  const payload = {
    contents: [{ parts: [{ text: prompt }] }],
    generationConfig: { temperature: 0.1, maxOutputTokens: 256 }
  };

  const res = UrlFetchApp.fetch(url, {
    method: 'post',
    contentType: 'application/json',
    payload: JSON.stringify(payload),
    muteHttpExceptions: true
  });

  const body = JSON.parse(res.getContentText());
  if (body.error) throw new Error('Gemini API エラー: ' + body.error.message);

  const raw = body.candidates[0].content.parts[0].text;
  const match = raw.match(/\{[\s\S]*\}/);
  if (!match) throw new Error('Gemini のレスポンスをJSONとして解析できませんでした');

  return JSON.parse(match[0]);
}
