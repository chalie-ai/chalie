# Chalie Web Interface Specification

The Chalie web interface is a single-page chat application that allows users to interact with the cognitive assistant. It communicates with the backend API via REST endpoints.

## Design Requirements

1. **Simple and clean design** using Bootstrap as the base framework
2. **Dark color scheme** following UX best practices (dark gray background #1a1a1a, orange accents #ff6b35)
3. **100% responsive** — Mobile-first design that scales to desktop
4. **Single-page application (SPA)** — No page reloads
5. **Consistent branding** — "Chalie" header and color scheme throughout
 
## Layout

### Page Structure
1. **Title Bar** (fixed at top, ~60px height)
   - App name "Chalie" centered in orange
   - Optional: Media player on the right (shown when audio is playing)

2. **Chat Area** (scrollable middle section)
   - Messages displayed in a ChatGPT/Messenger style layout
   - System messages (Chalie) aligned left with light background
   - User messages aligned right with orange/accent color background
   - Support for multiple consecutive messages from either party

3. **Prompt Box** (fixed at bottom, ~80px height)
   - Text input field for user messages
   - Left side: Microphone button (voice input)
   - Right side: Send button (arrow icon)
   - Visual feedback while message is being sent

### Responsive Design
- **Mobile**: Full-width, touch-friendly buttons, optimized for portrait orientation
- **Tablet**: Increased spacing, readable chat width
- **Desktop**: Centered chat area with max width, comfortable spacing

## Features

### Input Methods

#### Text Input
- Type messages in the prompt box at the bottom
- Click the send button (arrow icon) to submit
- Visual feedback indicates message is being sent
- Prompt box clears automatically after sending

#### Voice Input (Optional)
- Click the microphone button to start recording
- Screen locks to prevent accidental interactions while recording
- Speech-to-text converts voice to text
- Transcribed text appears in the prompt box
- User must click send button to submit (not automatic)

### Message Display

#### User Messages
- Displayed on the right side of the chat
- Orange/accent color background
- Timestamp optional

#### Chalie Messages
- Displayed on the left side of the chat
- Light background (contrast with dark theme)
- Optional speaker icon below message for text-to-speech playback

### Text-to-Speech (Optional)

When available:
- Speaker icon appears below Chalie messages
- Click to open audio player in title bar
- Audio player shows: play/pause controls and close button
- Audio plays immediately when opened
- Allows user to listen to responses while reading or multitasking