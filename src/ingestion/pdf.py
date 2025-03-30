from src.llm.gemini import gemini_2flash
import httpx
from PyPDF2 import PdfReader, PdfWriter
import re
import io
from difflib import SequenceMatcher

prompt = """You are an accurate AI system that extracts PDF content into clean, readable markdown.
    
    Current task:
    - Processing PDF pages {start_page} to {end_page}
    - One page overlaps with previous batch for continuity
    
    Previous batch context:
    <previous_batch_markdown>
    {previous_batch_markdown}
    </previous_batch_markdown>
    
    MAIN GOAL: Create continuous, readable markdown text that flows naturally.
    PRIORITY: Prioritize readability and flow above all else, simplifying the structure of the text when necessary.
    IMPORTANT: Ensure that there are no hallucinations in the text, your output MUST NOT add any information that is not present in the input.
    
    FORMATTING GUIDELINES:
    
    1. DOCUMENT STRUCTURE
        - Convert chapter titles to # (h1)
        - Convert section headers to ## (h2)
        - Convert subsections to ### (h3)
        - Base hierarchy on semantic function, not visual formatting
        - If the previous batch ended mid-sentence, start the new batch at the beginning of the last sentence and continue from there
    
    2. TEXT BLOCKS
        - Format quotes and poems with > blockquotes
        - Keep paragraphs as continuous text - never break mid-paragraph
        - Only create new lines at true paragraph endings (after periods, question marks, etc.)
    
    3. SPECIAL ELEMENTS
        - Format footnotes as [^page_number-index]
        - Convert bulleted/numbered lists to proper markdown lists ONLY if there are more than 2 items
        - Maintain nested list indentation
    
    4. CLEANUP RULES
        - Remove page numbers, headers, footers, and watermarks that are not part of the content
        - Skip non-content elements (headers, footers, etc.) that interrupt reading flow
        - Never use ```markdown``` blocks in your output
        - If previous batch ended mid-sentence, restart at beginning of that sentence
    
    PROCESS:
    1. First, reflect on content structure between <reflection></reflection> tags to determine what would be the most readable way to format the content according to the formatting guidelines.
    2. Note potential pitfalls between <understanding></understanding> tags, that could be problematic for the readability of the text.
    3. Create initial markdown between <draft></draft> tags, following the formatting guidelines.
    4. Self-critique your work between <critique></critique> tags, focusing on the readability and flow of the text. Check that there are no hiccups in the flow of the text. Ensure that there are no hallucinations in the text.
    5. Provide final markdown between <final></final> tags, implementing the critique and following the formatting guidelines.

    If the pages are empty, just return an empty string between <final></final> tags.
    REMEMBER: Prioritize readability and flow above all else. The result should read like a continuous document with no artificial breaks."""
        


def crop_pdf_page(page, top_offset_percent: float = 0, bottom_offset_percent: float = 0):
    """
    Crop a PDF page by removing portions from the top and bottom based on percentage of page height.
    
    Args:
        page: The PDF page object to crop
        top_offset_percent: Percentage of page height to crop from the top (0-100)
        bottom_offset_percent: Percentage of page height to crop from the bottom (0-100)
        
    Returns:
        The cropped PDF page object
    """
    # Get the current media box (page dimensions)
    media_box = page.mediabox
    
    # Calculate the actual offsets based on percentages
    page_height = float(media_box.upper_right[1]) - float(media_box.lower_left[1])
    top_offset = (top_offset_percent / 100) * page_height
    bottom_offset = (bottom_offset_percent / 100) * page_height
    
    # Apply the crop by modifying the media box
    new_lower_left_y = float(media_box.lower_left[1]) + bottom_offset
    new_upper_right_y = float(media_box.upper_right[1]) - top_offset
    
    # Create the new media box coordinates
    page.mediabox.lower_left = (float(media_box.lower_left[0]), new_lower_left_y)
    page.mediabox.upper_right = (float(media_box.upper_right[0]), new_upper_right_y)
    
    return page


async def test_crop_pdf_page(pdf_url: str, top_crop_percent: float = 0, bottom_crop_percent: float = 0):
    # Download the PDF content
    doc_data = httpx.get(pdf_url).content
    
    # Create a PDF reader object from the bytes
    pdf_reader = PdfReader(io.BytesIO(doc_data))

    # Create a PDF writer for the output
    pdf_writer = PdfWriter()
    
    # Process pages 10 through 25
    for page_num in range(10, 26):
        page = pdf_reader.pages[page_num]
        cropped_page = crop_pdf_page(page, top_crop_percent, bottom_crop_percent)
        pdf_writer.add_page(cropped_page)
    
    # Write the processed pages to bytes buffer
    output_buffer = io.BytesIO()
    pdf_writer.write(output_buffer)
    output_buffer.seek(0)
    
    # Save the PDF bytes to a local file for testing/debugging
    output_bytes = output_buffer.getvalue()
    with open("cropped_output.pdf", "wb") as f:
        f.write(output_bytes)


def merge_markdown_batches(previous: str, current: str, *, min_overlap: int = 16) -> str:
    """
    Merge two markdown batches by finding and resolving the overlap between the end of previous
    and beginning of current text
    """
    if not previous:
        return current

    # Use SequenceMatcher to find matching blocks
    matcher = SequenceMatcher(None, previous, current)
    matches = matcher.get_matching_blocks()
    
    # Filter for matches that are:
    # 1. Long enough (min_overlap)
    # 2. At the end of the previous text (a + size == len(previous))
    valid_matches = [
        m for m in matches 
        if m.size >= min_overlap and (m.a + m.size == len(previous))
    ]
    
    # No valid overlap at end of previous text, append with separator
    if not valid_matches:
        return "\n" + current
    
    # Get the longest match at the end of previous text
    best_match = max(valid_matches, key=lambda m: m.size)
    print("# Best match::: ", current[:best_match.b])
    
    # Merge using the found overlap
    # We take all of previous and append current starting after the overlap
    return current[best_match.b + best_match.size:]


def is_special_markdown_line(line: str) -> bool:
    """
    Determines if a line is a special markdown element like header, quote, list item, etc.
    
    Args:
        line: The line to check
        
    Returns:
        True if the line is a special markdown element, False otherwise
    """
    return (not line or 
           line.startswith(('#', '>', '[^', '-', '*')) or 
           re.match(r'^\d+\.', line) or 
           line.endswith(('.', '!', '?')))


def is_header(line: str) -> bool:
    """
    Determines if a line is a markdown header.
    
    @param line: The line to check
    @return True if the line is a header, False otherwise
    """
    return line.startswith('#')


def is_complete_paragraph(text: str) -> bool:
    """
    Determines if a paragraph is complete (ends with punctuation).
    
    @param text: The paragraph to check
    @return True if the paragraph is complete, False otherwise
    """
    return text.endswith(('.', '!', '?'))


def clean_markdown_paragraphs(markdown_content: str) -> str:
    """
    Cleans markdown content by correctly formatting paragraphs.
    
    This function:
    1. Maintains special markdown elements (headers, lists, etc.)
    2. Joins lines that belong to the same paragraph
    3. Handles incomplete paragraphs
    
    @param markdown_content: The raw markdown content
    @return Cleaned markdown with properly formatted paragraphs
    """
    lines = markdown_content.splitlines()
    current_paragraph, cleaned_lines = "", []
    
    # First pass: join paragraph lines
    for i, line in enumerate(lines):
        line = line.strip()

        # We don't tollerate headers that are too long
        if is_header(line) and len(line) > 100:
            line = line.replace('#', '').strip()
        
        # If current line is special, flush any accumulated paragraph and add the line
        if is_special_markdown_line(line):

            # Clean the current paragraph, if present
            if current_paragraph:
                cleaned_lines.append(current_paragraph)
                current_paragraph = ""

            # Always add the special line   
            cleaned_lines.append(line)
            continue

        # If this is the first line of a new paragraph, or if the current paragraph is empty, add the line
        current_paragraph = (current_paragraph + " " + line) if current_paragraph else line

        # If the current paragraph is complete, add it to the cleaned lines
        if is_complete_paragraph(current_paragraph):
            cleaned_lines.append(current_paragraph)
            current_paragraph = ""
    
    # Add any remaining paragraph
    current_paragraph and cleaned_lines.append(current_paragraph)

    # Post-processing to handle headers after incomplete paragraphs
    # This step eliminates headers that appear after paragraphs not properly closed
    # and before another paragraph (indicating the header might be misplaced content)
    filtered_lines = []
    i = 0
    while i < len(cleaned_lines):
        line = cleaned_lines[i]
        
        # Check for pattern: incomplete paragraph + header + paragraph
        if (i < len(cleaned_lines) - 2 and
            not is_special_markdown_line(line) and 
            not is_complete_paragraph(line) and
            is_header(cleaned_lines[i+1]) and
            not is_special_markdown_line(cleaned_lines[i+2])):
            
            # Skip the header and merge the paragraphs
            filtered_lines.append(line + " " + cleaned_lines[i+2])
            i += 3  # Skip the two lines we just processed plus current
            continue

        # Otherwise, add the line
        filtered_lines.append(line)
        i += 1
    
    return '\n'.join(filtered_lines)


async def convert_pdf_to_markdown(
    pdf_url: str, 
    *, 
    batch_size: int = 2, 
    overlap_pages: int = 0,
    top_crop_percent: float = 0,
    bottom_crop_percent: float = 0
) -> str:
    """
    Convert PDF to markdown using overlapping batches and context-aware processing.
    Writes progress to temp.txt for monitoring.
    
    @param pdf_url: URL of the PDF to process
    @param batch_size: Number of pages to process in each batch
    @param overlap_pages: Number of pages to overlap between batches
    @param top_crop_percent: Percentage of page height to crop from the top (0-100)
    @param bottom_crop_percent: Percentage of page height to crop from the bottom (0-100)
    
    @return The complete markdown content
    """

    # Create a temp file to store the markdown
    filename = pdf_url.split('/')[-1].replace('.', ' ')
    with open(f"{filename}.md", 'w', encoding='utf-8') as f:
        f.write("")

    # Download the PDF content
    doc_data = httpx.get(pdf_url).content
    
    # Create a PDF reader object from the bytes
    pdf_reader = PdfReader(io.BytesIO(doc_data))
    total_pages = len(pdf_reader.pages)
    
    # Process the PDF in batches
    previous_batch_markdown = ""
    
    for start_page in range(0, total_pages, batch_size - overlap_pages):

        # Calculate the end page for this batch
        end_page = min(start_page + batch_size, total_pages)

        print(f"\n\n----- Processing batch {start_page + 1} to {end_page} -----")
        
        # Create a new PDF with just this batch of pages
        output = io.BytesIO()
        writer = PdfWriter()
        
        # Add the pages for this batch
        for page_num in range(start_page, end_page):
            page = pdf_reader.pages[page_num]

            # If cropping is requested, crop the page
            if top_crop_percent > 0 or bottom_crop_percent > 0:
                page = crop_pdf_page(page, top_crop_percent, bottom_crop_percent)

            # Add the cropped page to the writer
            writer.add_page(page)
        
        # Write the batch to the output buffer
        writer.write(output)
        batch_bytes = output.getvalue()
        
        # Compile the prompt with the current batch information
        compiled_prompt = prompt.format(start_page=start_page, end_page=end_page, previous_batch_markdown=previous_batch_markdown)

        for _ in range(8):
            try:
                current_batch_output = await gemini_2flash.run(compiled_prompt, pdf_bytes=[batch_bytes])
                current_batch_parsed = re.search(r'<final>(.*?)</final>', current_batch_output, re.DOTALL)
                current_batch_markdown, copied = (current_batch_parsed.group(1).strip() for _ in range(2))
                break
            except Exception as e:
                print(f"# Error: {e}")
                print(f"# The output was: {current_batch_output}")
                continue

        # Remove markdown code blocks (that should not be present in the final markdown)
        current_batch_markdown = re.sub(r'```markdown', '', current_batch_markdown)
        current_batch_markdown = re.sub(r'```', '', current_batch_markdown)

        # Clean paragraphs in the markdown content
        current_batch_markdown = clean_markdown_paragraphs(current_batch_markdown)

        # Merge with previous batch and update final markdown
        current_batch_markdown = merge_markdown_batches(previous_batch_markdown, current_batch_markdown)

        # Write current state to temp file
        with open(f"{filename}.md", 'a', encoding='utf-8') as f:
            f.write(current_batch_markdown)
        
        with open(f"{filename}-logs.txt", 'a', encoding='utf-8') as f:
            f.write(f"# Parsing made the batch ({start_page} - {end_page}) go: {len(copied)} -> {len(current_batch_markdown)}\n")

        previous_batch_markdown = current_batch_markdown
