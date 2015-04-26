import re

"""
The purpose of this file is to 'clean' the emails
of anything but the actual main body of the email.
This includes removing 'original messages' and tags
that are in the header of each email file.
"""


def format_email(email):
	email = remove_tags(email)
	email = remove_forwards(email)
	email = remove_replies(email)
	#remove markup on the emails
	email = re.sub(r"<.*>", r"", email)
	return email


def remove_tags(email):
	# each email header ends with this tag
	xfile_name_pat = re.compile("X-FileName:.*\n")

	# split at final tag
	email = re.split(xfile_name_pat, email, 1)

	# made this safe in case there was no split
	email = email[len(email) - 1]

	return email


def remove_forwards(email):
	"""
	Purpose of this method is to remove forwarded emails from
	the email file sent in.
	"""
	forward_pattern = re.compile("---------------------- Forwarded by.+---------------------------", re.DOTALL)
	forward_patterns = [forward_pattern]

	forward_pattern = re.compile("----- Forwarded by.+ -----")
	forward_patterns.append(forward_pattern)

	# use below to add when other formats are found
	# forward_pattern = re.compile("this")
	# forward_patterns.append(forward_pattern)

	for forward_pat in forward_patterns:
		email = re.split(forward_pat, email, 1)[0]

	return email


def remove_replies(email):
	"""
	Removes the emails that are listed below what the user actually
	sent (i.e. messages they've replied to or the 'Original Message').
	Note: remove tags should be used first or this may misbehave
	"""
	# string formats

	s_replies_formats = ['-----Original Message-----']

	# s_replies_formats.append('new format')

	for reply_format in s_replies_formats:
		email = email.split(reply_format, 1)[0]

	# regular expressions formats

	# this pattern is modeled after those like seen in 'arnold-j/sent/17 and 10' aka internal
	# Note the range limits to capture the right part of the email and not let it span outside
	reply_pattern = re.compile("\n.{1,30}@.{1,30}\n{1,2}.{1,5}/.{1,2}/.{1,20}\n{1,2}.{1,10}To:.*\n.{0,10}cc:.*\n.{0,10}Subject:")
	re_replies_formats = [reply_pattern]

	# modeled after 'arnold-j/sent/3' aka external
	reply_pattern = re.compile("\n.{1,30}@.{1,30}/.{1,3}/.{1,20}\nTo:.*\ncc:.*\nSubject:")
	re_replies_formats.append(reply_pattern)

	# modeled after 'arora-h/sent/6 aka 'From:' style
	# reply_pattern = re.compile("\nFrom:.{1,30}@.{1,40}/.{1,2}/.{1,20}To:.*cc:.*Subject:", re.DOTALL)
	# re_replies_formats.append(reply_pattern)

	reply_pattern = re.compile("\n.{0,4}From:.{1,70}/.{1,2}/.{1,20}To:.*cc:.*Subject:", re.DOTALL)
	re_replies_formats.append(reply_pattern)

	# modeled after 'bass-e/sent/10 aka 'To:' style
	reply_pattern = re.compile("\nTo:.*cc:.*Subject:", re.DOTALL)
	re_replies_formats.append(reply_pattern)

	# modeled after dasovic-j/sent/1024 aka =09 in front of each line of the reply
	reply_pattern = re.compile("\n.{1,40}@.{1,22}\n{1,2}.{1,6}/.{1,3}/.{1,32}To:.*\n.{0,10}cc:.*\n.{0,10}Subject:", re.DOTALL)
	re_replies_formats.append(reply_pattern)

	# modeled after bass-e/sent/330 aka external list style
	reply_pattern = re.compile("\nContent-transfer-encoding: .{1,10}\n")
	re_replies_formats.append(reply_pattern)

	for reply_pat in re_replies_formats:
		email = re.split(reply_pat, email, 1)[0]

	return email